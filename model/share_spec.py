import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from transformers import BertModel, BertConfig

# from utils import to_gpu
# from utils import ReverseLayerF
from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from scipy.signal import stft, istft
import torchaudio



"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""
seed = 42
random.seed(seed)  # 设置 Python 随机种子
np.random.seed(seed)  # 设置 NumPy 随机种子
torch.manual_seed(seed)  # 设置 PyTorch 随机种子
torch.cuda.manual_seed(seed)  # 设置当前 GPU 随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

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



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss




# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self):
        super(MISA, self).__init__()

        self.output_size = output_size = 3
        self.dropout_rate = dropout_rate = 0.5
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=1024, out_features=64))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(64))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=1024, out_features=64))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(64))

        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=64, out_features=64))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=64, out_features=64))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=64, out_features=64))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=64, out_features=64))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=64, out_features=64))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=64*4, out_features=64*2))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=64*2, out_features= 64))

        self.fusion1 = nn.Sequential()
        self.fusion1.add_module('fusion_layer_1', nn.Linear(in_features=64*3, out_features=64*2))
        self.fusion1.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion1.add_module('fusion_layer_1_activation', self.activation)
        self.fusion1.add_module('fusion_layer_3', nn.Linear(in_features=64*2, out_features= 64))

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.text_weight = nn.Parameter(torch.randn(64,64, dtype=torch.float32))
        self.image_weight = nn.Parameter(torch.randn(64,64, dtype=torch.float32))

        
    def js_div(self,p, q):
        """
        Function that measures JS divergence between target and output logits:
        """
        M = (p + q) / 2
        kl1 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        gamma = 0.5 * kl1 + 0.5 * kl2
        return gamma


    def alignment(self, sentences, visual):
       
        self.reconstruct()
      
        alpha = self.js_div(self.utt_shared_t, self.utt_shared_v)
        a=torch.matmul(self.utt_shared_t, self.text_weight)

        b=torch.matmul(self.utt_shared_v, self.image_weight)
        f=(1-alpha) * (a+b) + alpha * self.utt_shared_t + alpha * self.utt_shared_v

        h = torch.stack((self.utt_private_t, self.utt_private_v, f), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2]), dim=1)
        o = self.fusion1(h)

        return o
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)


    def shared_private(self, utterance_t, utterance_v):

        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)

        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)

    def forward(self, sentences, video ):

        o = self.alignment(sentences, video)

        return o


class IMFM(nn.Module):
    def __init__(self):
        super(IMFM, self).__init__()

        self.output_size = output_size = 3
        self.dropout_rate = dropout_rate = 0.5
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        # Project ECG and CXR into a common space
        self.project_ecg = nn.Sequential(
            nn.Linear(in_features=1024, out_features=64),
            self.activation,
            nn.LayerNorm(64)
        )

        self.project_cxr = nn.Sequential(
            nn.Linear(in_features=1024, out_features=64),
            self.activation,
            nn.LayerNorm(64)
        )

        # Private layers for ECG and CXR
        self.private_ecg = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.Sigmoid()
        )

        self.private_cxr = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.Sigmoid()
        )

        # Shared layers for ECG and CXR
        self.shared = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.Sigmoid()
        )

        # Reconstruction layers for ECG and CXR
        self.recon_ecg = nn.Sequential(
            nn.Linear(in_features=64, out_features=64)
        )
        self.recon_cxr = nn.Sequential(
            nn.Linear(in_features=64, out_features=64)
        )

        # Fusion layers for ECG and CXR features
        self.fusion = nn.Sequential(
            nn.Linear(in_features=64*4, out_features=64*2),
            nn.Dropout(dropout_rate),
            self.activation,
            nn.Linear(in_features=64*2, out_features=64)
        )

        self.fusion1 = nn.Sequential(
            nn.Linear(in_features=64*3, out_features=64*2),
            nn.Dropout(dropout_rate),
            self.activation,
            nn.Linear(in_features=64*2, out_features=64)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Parameters for attention weights
        self.ecg_weight = nn.Parameter(torch.randn(64, 64, dtype=torch.float32))
        self.cxr_weight = nn.Parameter(torch.randn(64, 64, dtype=torch.float32))

    def js_div(self, p, q):
        """
        Function that measures JS divergence between target and output logits:
        """
        M = (p + q) / 2
        kl1 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        gamma = 0.5 * kl1 + 0.5 * kl2
        return gamma

    def alignment(self, ecg, cxr):
        self.reconstruct()

        alpha = self.js_div(self.utt_shared_ecg, self.utt_shared_cxr)
        a = torch.matmul(self.utt_shared_ecg, self.ecg_weight)
        b = torch.matmul(self.utt_shared_cxr, self.cxr_weight)
        f = (1 - alpha) * (a + b) + alpha * self.utt_shared_ecg + alpha * self.utt_shared_cxr

        h = torch.stack((self.utt_private_ecg, self.utt_private_cxr, f), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2]), dim=1)
        o = self.fusion1(h)

        return o

    def reconstruct(self):
        # Reconstruct the ECG and CXR from private and shared parts
        self.utt_ecg = self.utt_private_ecg + self.utt_shared_ecg
        self.utt_cxr = self.utt_private_cxr + self.utt_shared_cxr

        self.utt_ecg_recon = self.recon_ecg(self.utt_ecg)
        self.utt_cxr_recon = self.recon_cxr(self.utt_cxr)

    def shared_private(self, ecg_data, cxr_data):
        # Project the ECG and CXR data into common space
        self.utt_ecg_orig = ecg_data = self.project_ecg(ecg_data)
        self.utt_cxr_orig = cxr_data = self.project_cxr(cxr_data)

        # Generate private and shared representations
        self.utt_private_ecg = self.private_ecg(ecg_data)
        self.utt_private_cxr = self.private_cxr(cxr_data)

        self.utt_shared_ecg = self.shared(ecg_data)
        self.utt_shared_cxr = self.shared(cxr_data)

    def forward(self, ecg_data, cxr_data):
        self.shared_private(ecg_data, cxr_data)
        o = self.alignment(ecg_data, cxr_data)

        return o
