import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from scipy.signal import stft, istft
import torchaudio
#-----------------------this was adaptive from 
class ImagePatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d_model=256, in_channels=3):
        super(ImagePatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)  # (img_size, img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv_layer = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        B, C, H, W = image.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        image = self.conv_layer(image).flatten(2).transpose(1, 2)  # (B, H*W, D)
        return image

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),
                                          nn.Dropout(dropout),
                                          nn.GELU(),
                                          nn.Linear(d_ff, d_model),
                                          nn.Dropout(dropout))

    def forward(self, x):
        return self.feed_forward(x)

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super(AddNorm, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(d_model, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x)
        x_ = x
        x = self.dropout(x)
        x = self.feed_forward(x) + x_
        x = self.norm2(x)
        return x


# class Image2TextGate(nn.Module):
#     def __init__(self, n, d_model):
#         super(Image2TextGate, self).__init__()
#         self.n = n
#         self.avg_pool = nn.AvgPool1d(kernel_size=n)
#         self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
#         self.select_para = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))

#     def forward(self, image):
#         B, N, C = image.shape
#         assert N == self.n
#         image = image * torch.view_as_complex(self.select_para)
#         image = image.permute(0, 2, 1)  # (B, C, N)
#         image = self.avg_pool(image.real)  # (B, C, 1)
#         image = self.conv_layer(image)  # (B, C, 1)
#         image = image.permute(0, 2, 1)  # (B, 1, C)
#         return image

class Cxr2EcgGate(nn.Module):
    def __init__(self, spatial_size, model_dim):
        super(Cxr2EcgGate, self).__init__()

        # Parameters for image (CXR) to text (ECG) transformation
        self.spatial_size = spatial_size
        self.model_dim = model_dim

        # Learnable selection parameters and layers
        self.selection_parameters = nn.Parameter(torch.randn(spatial_size, model_dim, 2, dtype=torch.float32))
        self.pool_layer = nn.AvgPool1d(kernel_size=spatial_size)
        self.convolution_layer = nn.Conv1d(model_dim, model_dim, kernel_size=1)

    def _apply_selection(self, input_data):
        """
        Apply element-wise selection and transformation on the input data.
        """
        return input_data * torch.view_as_complex(self.selection_parameters)

    def forward(self, cxr_input):
        # Ensure input dimensions match
        B, N, C = cxr_input.shape
        assert N == self.spatial_size, f"Expected spatial size {self.spatial_size}, but got {N}"

        # Apply the selection and transformation
        selected_cxr = self._apply_selection(cxr_input)

        # Rearrange for convolution (change to (B, C, N))
        selected_cxr = selected_cxr.permute(0, 2, 1)

        # Pooling and convolution operations
        pooled_cxr = self.pool_layer(selected_cxr.real)  # Perform pooling over the real part
        convolved_cxr = self.convolution_layer(pooled_cxr)

        # Permute back to original format (B, 1, C)
        return convolved_cxr.permute(0, 2, 1)



# class Text2ImageGate(nn.Module):
#     def __init__(self, s, d_model):
#         super(Text2ImageGate, self).__init__()
#         self.s = s
#         self.avg_pool = nn.AvgPool1d(kernel_size=s)
#         self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
#         self.select_para = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))

#     def forward(self, text):
#         text = text * torch.view_as_complex(self.select_para)  # (B, S, C)
#         text = text.permute(0, 2, 1)
#         text = self.avg_pool(text.real)  # (B, C, 1)
#         text = self.conv_layer(text)  # (B, C, 1)
#         text = text.permute(0, 2, 1)  # (B, 1, C)
#         return text

class Ecg2CxrGate(nn.Module):
    def __init__(self, spatial_size, model_dim):
        super(Ecg2CxrGate, self).__init__()

        # Parameters for text (ECG) to image (CXR) transformation
        self.spatial_size = spatial_size
        self.model_dim = model_dim

        # Learnable selection parameters and layers
        self.selection_parameters = nn.Parameter(torch.randn(spatial_size, model_dim, 2, dtype=torch.float32))
        self.pool_layer = nn.AvgPool1d(kernel_size=spatial_size)
        self.convolution_layer = nn.Conv1d(model_dim, model_dim, kernel_size=1)

    def _apply_selection(self, input_data):
        """
        Apply element-wise selection and transformation on the input data.
        """
        return input_data * torch.view_as_complex(self.selection_parameters)

    def forward(self, ecg_input):
        # Ensure input dimensions match
        B, S, C = ecg_input.shape
        assert S == self.spatial_size, f"Expected spatial size {self.spatial_size}, but got {S}"

        # Apply the selection and transformation
        selected_ecg = self._apply_selection(ecg_input)

        # Rearrange for convolution (change to (B, C, S))
        selected_ecg = selected_ecg.permute(0, 2, 1)

        # Pooling and convolution operations
        pooled_ecg = self.pool_layer(selected_ecg.real)  # Perform pooling over the real part
        convolved_ecg = self.convolution_layer(pooled_ecg)

        # Permute back to original format (B, 1, C)
        return convolved_ecg.permute(0, 2, 1)


class ecg_gate(nn.Module):
    def __init__(self, s, d_model):
        super(ecg_gate, self).__init__()

        self.text_gate = Ecg2CxrGate(s, d_model)

    def forward(self, image, text):
        """
        image: (B, N, C)  N=h*w  in frequency domain
        """
        text_gate = self.text_gate(text)
        image = image * text_gate
        return image

class cxr_gate(nn.Module):
    def __init__(self, n, d_model):
        super(cxr_gate, self).__init__()

        self.image_gate = Cxr2EcgGate(n, d_model)

    def forward(self, text, image):
        image_gate = self.image_gate(image)
        text = text * image_gate
        return text

pi=3.1415926

class FtLayer(nn.Module):
    def __init__(self, d_model, s, n, num_filter=2, dropout=0.,use_bank=True):
        super(FtLayer, self).__init__()
        self.s = s
        self.n = n
        self.use_bank = use_bank
        self.num_filter = num_filter

        self.ecg_weight = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))
        self.ecg_filter_bank = nn.Parameter(torch.randn(num_filter, s, d_model, 2, dtype=torch.float32))

        self.cxr_weight = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))
        self.cxr_filter_bank = nn.Parameter(torch.randn(num_filter, n, d_model, 2, dtype=torch.float32))

        self.ecg_frequency_select = cxr_gate(n, d_model)
        self.cxr_frenquency_select = ecg_gate(s, d_model)

        self.ecg_add_norm = AddNorm(d_model, dropout)
        self.cxr_add_norm = AddNorm(d_model, dropout)

    def filter(self, x, length, filter_bank, weight):
        if self.use_bank:
            # print(x.device)
            power = (x * x) / length
            # print(power.device)#cpu
            # print(power.device, filter_bank[0].device, cos.device)
            Y = []
            for k in range(self.num_filter):
                cos = torch.cos(torch.as_tensor((2 * (k + 1) - 1) * pi / 2 * self.num_filter))
                Y.append(power * filter_bank[k] * cos)
               

            C = torch.stack(Y)  # (filter, batch, s, dim)
            x = torch.sum(C, dim=0)  # (batch, s, dim)
        else:
            x = x * weight

        return x

    def forward(self, ecg, image, spatial_size=None):
        x_ecg = ecg#[B,12,4096]
        initial_ecg=ecg

        B, S, D = ecg.shape

        x_image = image
        B, N, C = image.shape #should be [B,224*224,3]

        assert N // 2 + 1 == self.n #n should be 25089

        ecg1=ecg.detach().cpu().numpy()
        ecg1=ecg1.transpose(0,2,1)

        f,t, _ecg = stft(ecg1,fs=400, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]

        _ecg=_ecg.transpose(0,2,3,1)

        a,b,c,d=_ecg.shape
        _ecg=_ecg.reshape(a,b*c,d)
        B,S,D=_ecg.shape
        assert S  == self.s

        _ecg = torch.tensor(_ecg, dtype=torch.complex64)
        ecg_device = ecg.device  # 获取 ecg 的设备
        _ecg = _ecg.to(ecg_device)

        _image = torch.fft.rfft(image, dim=1, norm='ortho') #complex
        real_part = self.ecg_filter_bank[..., 0]
        imag_part = self.ecg_filter_bank[..., 1]
        complex_tensor = torch.complex(real_part, imag_part)
        real_part = self.ecg_weight[..., 0]
        imag_part = self.ecg_weight[..., 1]
        complex_weight = torch.complex(real_part, imag_part)
        _ecg = self.filter(_ecg, self.s, complex_tensor,
                            complex_weight)
        _image = self.filter(_image, self.n, torch.view_as_complex(self.cxr_filter_bank.contiguous()),
                             torch.view_as_complex(self.cxr_weight))
        is_complex = _ecg.is_complex()
        # print(f"Is the 1 ECG data complex? {is_complex}")
        is_complex = _image.is_complex()

        ecg = self.ecg_frequency_select(_ecg, _image)
        image = self.cxr_frenquency_select(_image, _ecg)

        image = torch.fft.irfft(_image, n=N, dim=1, norm='ortho')
        image = image.view(B, N, C)
        ecg=ecg.detach().cpu().numpy()

        ecg=ecg.reshape(a,b,c,d)#[batch,257,17,128]
        ecg=ecg.transpose(0,3,1,2)
        # print(f'after reshape stft ecg {ecg.shape}')
        _,ecg = istft(ecg,fs=400, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]
        ecg = torch.tensor(ecg)
        ecg_device = initial_ecg.device  # 获取 ecg 的设备
        ecg = ecg.to(ecg_device)

        ecg=ecg.permute(0,2,1)

        text = self.ecg_add_norm(ecg + x_ecg)
        image = self.cxr_add_norm(image + x_image)

        return text, image

#-------------

class FtBlock(nn.Module):
    def __init__(self, d_model, s, n, num_layer=1, num_filter=2, dropout=0.):
        """
        :param d_model:
        :param s: seq_len / 2 + 1
        :param h:
        :param w:
        :param n:
        """
        super(FtBlock, self).__init__()
        self.ft = nn.ModuleList([FtLayer(d_model, s, n, num_filter, dropout) for _ in range(num_layer)])

    def forward(self, text, image):
        for ft_layer in self.ft:
            text, image = ft_layer(text, image)

        return text, image
