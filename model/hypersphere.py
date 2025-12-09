import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from scipy.signal import stft, istft
import torchaudio
import sys
import os
# sys.path.append(os.path.abspath('/home/ke/MIMIC_subset/MIMIC_subset'))
from torch.nn import Parameter
from model.fusion_model import ImagePatchEmbed
from model.utils import func_attention,Router,Refinement
from model.xlstm_used import xLSTM
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
# from .loss import RankingLoss, CosineLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
# from argument import args_parser
# parser = args_parser()
# add more arguments here ...
# args = parser.parse_args()



import torch.nn as nn
import numpy as np

#========senet===============
from torch import nn
import random
seed = 42
random.seed(seed)  # 设置 Python 随机种子
np.random.seed(seed)  # 设置 NumPy 随机种子
torch.manual_seed(seed)  # 设置 PyTorch 随机种子
torch.cuda.manual_seed(seed)  # 设置当前 GPU 随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c ,_= x.size()

        y=self.avg_pool(x)
        # print(f'after avg y {y.shape}')#[bs,128,1]
        y=y.permute(0,2,1)
        y = self.fc(y).view(b, c, 1)
        # print(f'y {y.shape}')
        # x=x.permute(0,2,1)
        return x * y.expand_as(x)
#self.se = SELayer(planes, reduction)
#===================senet==============

#================resnet1d===================
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
        # if n_filters_out!=12:
            # print(f'12 != n_filters_out {n_filters_out}')
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        # print(f'Input x is on device: {x.device}, Input y is on device: {y.device}')

        if self.skip_connection is not None:
            # print(f'start skip_connection')
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        # print(f'start blk first conv1 {x.shape}')
        # print(f'conv1 weights are on device: {self.conv1.weight.device}')
        # print(f'conv2 weights are on device: {self.conv2.weight.device}')

        x = self.conv1(x)
        # print(f'after blk first conv1 {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        # print(f'after blk dropout1 x size {x.shape}')

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    def __init__(self,  n_classes=4, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        # self.blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh))
        self.blocks_dim=list(zip([64, 128, 196, 256, 320], [4096, 2048, 1024, 512, 256]))

        n_filters_in, n_filters_out = 12, self.blocks_dim[0][0]
        n_samples_in, n_samples_out = 4096, self.blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        # self.blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh))

        # Residual block layers
        # self.res_blocks = []
        self.res_blocks = nn.ModuleList()
        for i, (n_filters, n_samples) in enumerate(self.blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            
            # print(f'{i} th downsample {downsample}' )
            # print(f'i th n_filters_out {n_filters_out}')
            resblk1d = ResBlock1d( n_filters_in,n_filters_out, downsample, kernel_size, dropout_rate)
            # self.add_module('resblock1d_{0}'.format(i), resblk1d)
            # self.res_blocks += [resblk1d]
            self.res_blocks.append(resblk1d)

        n_filters_last, n_samples_last = self.blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, 512)

        self.lin1 = nn.Linear(512, 128)
        self.n_blk = len(self.blocks_dim)
        self.lin128 = nn.Linear(320, 128)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        # print(f' 1 input size {x.shape}')#[bs,12,4096]
        x = self.conv1(x)
        # print(f' 2 input size {x.shape}')
        x = self.bn1(x)
        # print(f' 3 input size {x.shape}')

        # Residual blocks
        y = x
        for i,blk in enumerate(self.res_blocks):
            # print(f' {i}th input shape {x.shape}')
            x, y = blk(x, y)
            # print(f' {i}th output x shape {x.shape}')
            # print(f' {i}th output y shape {y.shape}')

        # Flatten array
        # print(f'before flatten {x.shape}')#[bs,320,16]
        x=x.permute(0,2,1)
        x=self.lin128(x)
        # print(f'x {x.shape}')#[bs,16,128]
        # x = x.view(-1, 128, 16)  # 将输出调整回 [bs, 128, 16]
        # print(f'after view x {x.shape}')
        # x1 = x.view(x.size(0), -1)

        # # Fully conected layer
        # x2 = self.lin(x1)
        # # print(f'x2 {x2.shape}')#[bs,512]
        # x=self.lin1(x2)
        # # print(f'x {x.shape}')#[bs,128]
        return x






class CXRModels(nn.Module):
    def __init__(self):
        super(CXRModels, self).__init__()

        self.vision_backbone = torchvision.models.resnet34(pretrained=True)
        classifiers = ['classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity())
            break
            
        self.classifier = nn.Sequential(nn.Linear(d_visual, 128))
        self.feats_dim = d_visual

    def forward(self, x):
        # 获取 backbone 的输出
        visual_feats = self.vision_backbone.conv1(x)
        visual_feats = self.vision_backbone.bn1(visual_feats)
        visual_feats = self.vision_backbone.relu(visual_feats)
        visual_feats = self.vision_backbone.maxpool(visual_feats)

        visual_feats = self.vision_backbone.layer1(visual_feats)
        visual_feats = self.vision_backbone.layer2(visual_feats)
        visual_feats = self.vision_backbone.layer3(visual_feats)
        visual_feats = self.vision_backbone.layer4(visual_feats)


        # preds = self.classifier(visual_feats.view(visual_feats.size(0), -1))  # 展平
        return visual_feats

import torch
import torch.nn as nn
import math
#----Exchanging Dual-Encoder–Decoder: A New Strategy for Change Detection With Semantic Guidance and Spatial Localization--
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

class ECGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        #换为kernel_size=1
        # self.conv1 = nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0)
        #换为kernel_size=1

        # self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool = nn.AdaptiveMaxPool1d(output_size=16)

    def forward(self, ecg):
        ecg = self.conv1(ecg)  # [bs, 64, 128]
        ecg = self.pool(ecg)  # [bs, 64, 32]
        # ecg=nn.LayerNorm(ecg)
        return ecg

class CXRFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        #换为kernel_size=1
        # self.conv1 = nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0)
        #换为kernel_size=1
        # self.pool = nn.AdaptiveAvgPool1d(32)
        self.pool = nn.AdaptiveMaxPool1d(output_size=16)

    def forward(self, cxr):
        # print(f'cxr input {cxr.shape}')
        cxr = self.conv1(cxr)  # [bs, 64, 25]
        cxr = self.pool(cxr)   # [bs, 64, 32]
        # cxr=nn.LayerNorm(cxr)
        return cxr

class MLP(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim, num_class, act_layer=nn.ReLU, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_dim, outputs_dim)
        self.norm2 = nn.LayerNorm(outputs_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(outputs_dim, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act_layer(x)
        x = self.fc3(x)
        return x
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


class ECGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveMaxPool1d(output_size=16)

    def forward(self, ecg):
        ecg = self.conv1(ecg)  # [bs, 64, 128]
        ecg = self.pool(ecg)  # [bs, 64, 32]
        # ecg=nn.LayerNorm(ecg)
        return ecg

class CXRFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        #换为kernel_size=1
        # self.conv1 = nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0)
        #换为kernel_size=1
        # self.pool = nn.AdaptiveAvgPool1d(32)
        self.pool = nn.AdaptiveMaxPool1d(output_size=16)

    def forward(self, cxr):
        # print(f'cxr input {cxr.shape}')
        cxr = self.conv1(cxr)  # [bs, 64, 25]
        cxr = self.pool(cxr)   # [bs, 64, 32]
        # cxr=nn.LayerNorm(cxr)
        return cxr


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(64 * 32, 128)
        self.bn1 = nn.BatchNorm1d(128)  # 批归一化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout层，防止过拟合
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)  # 批归一化
        x = self.relu(x)
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)
        return x

def xavier_normal_(tensor, gain=1., mode='avg'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'avg':
        fan = fan_in + fan_out
    elif mode == 'in':
        fan = fan_in
    elif mode == 'out':
        fan = fan_out
    else:
        raise Exception('wrong mode')
    std = gain * math.sqrt(2.0 / float(fan))

    return nn.init._no_grad_normal_(tensor, 0., std)

class SFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, k=80.0, a=0.80, b=1.23):
        super(SFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.k = k
        self.a = a
        self.b = b

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        
        xavier_normal_(self.weight, gain=2, mode='out')

    def forward(self, input, label):
        # print(f'self.weight {self.weight.shape}')
        # print(f'input {input.shape}')
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        output = cosine * self.s

        loss_total = 0
        intra_total = 0
        inter_total = 0
        for i in range(3):
            label_i = label[:, i]  # [B]
            label_i = label_i.float()

            # one-hot for binary
            one_hot = label_i.view(-1, 1)  # [B, 1]
            zero_hot = 1.0 - one_hot       # [B, 1]

            output_i = output[:, i]  # [B]

            WyiX = one_hot.view(-1) * output_i  # [B]
            Wj = zero_hot.view(-1) * output_i  # [B]

            with torch.no_grad():
                theta_yi = torch.acos(WyiX / self.s)
                weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))

                theta_j = torch.acos(Wj / self.s)
                weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))

            intra_loss = -weight_yi * WyiX
            inter_loss = weight_j * Wj

            loss_i = intra_loss.mean() + inter_loss.mean()

            loss_total += loss_i
            intra_total += intra_loss.mean()
            inter_total += inter_loss.mean()

        # 最后求平均
        loss = loss_total / 3
        # print(f'loss {loss.shape}')
        return loss


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


# __all__ = ['align_loss', 'uniform_loss']

class hypershpere(nn.Module):
    def __init__(self,  d_text=12, seq_len=4369, img_size=224, patch_size=16, d_model=128,
                 num_filter=2, num_class=3, num_layer=1, dropout=0., mlp_ratio=4.):
        super(hypershpere, self).__init__()

        # Text

        self.text_encoder = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=d_model, kernel_size=17, stride=1, padding=8),  # 卷积层

            # nn.Linear(d_text, d_model),
                                        #   nn.LayerNorm(d_model),
                                          )
        # s = seq_len // 2 + 1
#=====未对ecg做rfft,所以特征维度不会变为1/2
        self.ecg_norm=nn.LayerNorm(d_model)
        s=seq_len


        # Image
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        # self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        # self.fusion = Fusion(d_model)

        self.mlp = MLP(d_model, int(mlp_ratio*d_model), d_model, num_class, dropout=dropout)
        self.mlp1 = MLP(64, int(mlp_ratio*64), 64, num_class, dropout=0.3)
        self.mlp2 = MLP(128, int(mlp_ratio*128), 128, num_class, dropout=0.3)
        self.mlp3 = MLP(9, int(mlp_ratio*9), 9, 9, dropout=0.3)
        # self.mlp1=Classifier(3)

        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

        self.resnet1d=ResNet1d()
        self.ecg_extractor = ECGFeatureExtractor()
        self.cxr_extractor = CXRFeatureExtractor()

        # config = CONFIGS['ViT-B_16']
        # self.vit=vit(config)
        # self.vit.load_from(np.load('/home/mimic/MIMIC_subset/MIMIC_subset/imagenet21k_ViT-B_16.npz'))
        # original_weights = np.load('/home/mimic/MIMIC_subset/MIMIC_subset/imagenet21k_ViT-B_16.npz')

        # 创建一个新的字典，剔除与 self.head 相关的权重
        # filtered_weights = {key: value for key, value in original_weights.items() if 'head' not in key}

        # 加载过滤后的权重
        # self.vit.load_from(filtered_weights)
        self.layernorm = nn.LayerNorm(128)
        self.act_layer = nn.ReLU()
        self.cxrmodel=CXRModels()
        self.se=SELayer(128,16)
        self.cxrlin=nn.Linear(512, 128)

        self.sfaceloss_e=SFaceLoss(in_features = 128, out_features=3, s = 64, k = 80, a = 0.8, b = 1.23)
        self.sfaceloss_c=SFaceLoss(in_features = 128, out_features=3, s = 64, k = 80, a = 0.8, b = 1.23)

        self.output_gate1 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Sigmoid()
            )
        self.output_gate2 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Sigmoid()
            )
        self.output1 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Tanh()
            )

        self.output2 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Tanh()
            )

#----------------------------
        self.output_gate3 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Sigmoid()
            )
        self.output_gate4 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Sigmoid()
            )
        self.output3 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Tanh()
            )
        self.output4 = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.Tanh()
            )
        self.conv_cxr1 = nn.Conv2d(512, 128, kernel_size=1)
        # self.cxr_fusion=AFF()
        # self.ecg_fusion=AFF()
        # self.vitlin=nn.Linear(768, 128)
        # self.vitlin2=nn.Conv1d(197, 49, kernel_size=1)
        # self.xLSTMLMModel= xLSTM(input_size=4096, head_size=1024, num_heads=2, batch_first=True, layers='ms')


# #---new added----
#         self.ecg_extractor = ECGFeatureExtractor()
#         self.cxr_extractor = CXRFeatureExtractor()


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            # nn.init.constant_(m.bias.data, 0.0)
            # trunc_normal_(m.weight, std=.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image,label):
        # print(f'input ecg {text.shape}')#[Batch,4096,12]
        text=text.permute(0,2,1)
        # print(f'input ecg {text.shape}')#[bs,12,4096]
#加入了resnet1d提取temporal ecg特征
#------resent1d--------------
        ecg_temporal=self.resnet1d(text)
        ecg_temporal=ecg_temporal.permute(0,2,1)

        # print(f'ecg_temporal {ecg_temporal.shape}')#[bs,128,256]
        ecg_temporal=self.act_layer(ecg_temporal)
        ecg_temporal=self.se(ecg_temporal)
        text = self.text_encoder(text)
        # print(f'after encoder ecg {text.shape}')
        text=text.permute(0,2,1)
        text=self.ecg_norm(text)

        image = image.to(torch.float32)

        cxr_spatial=self.cxrmodel(image)
        cxr_spatial=self.conv_cxr1(cxr_spatial)
        bs,c,h,w=cxr_spatial.shape
        cxr_spatial=cxr_spatial.view(bs,c,h*w)
        cxr_spatial=self.act_layer(cxr_spatial)
        cxr_spatial=self.se(cxr_spatial)


# #------new added-----
        ecg_temporal=self.ecg_extractor(ecg_temporal)
        cxr_spatial= self.cxr_extractor(cxr_spatial)
#先删掉extractor
        # print(f'after extractor ecg {ecg_temporal.shape}')#[bs,64,16]
        # print(f'after extractor cxr {cxr_spatial.shape}')#[bs,64,16]
        ecg_temporal = torch.max(ecg_temporal, dim=2)[0]
        cxr_spatial = torch.max(cxr_spatial, dim=2)[0]
        # print(f'after max ecg {ecg_temporal.shape}')#[bs,64]
        # print(f'after max cxr {cxr_spatial.shape}')#[bs,64]
        ecg_on_sphere = F.normalize(ecg_temporal, p=2, dim=1, eps=1e-12)
        cxr_on_sphere = F.normalize(cxr_spatial,   p=2, dim=1, eps=1e-12)
        # print(f'ecg_on_sphere {ecg_on_sphere.shape}')#[bs,64]
        # print(f'cxr_on_sphere {cxr_on_sphere.shape}')#[bs,64]
#先删掉
        # loss_e=self.sfaceloss_e(ecg_on_sphere,label)
        # loss_c=self.sfaceloss_c(cxr_on_sphere,label)
        # loo_sface=0.5*loss_e+0.5*loss_c
#先删掉
#-------add Sfaceloss----------------
        #-------add Sfaceloss----------------
        # alignloss=align_loss(ecg_on_sphere, cxr_on_sphere)

        # uni_ecg=uniform_loss(ecg_on_sphere)
        # uni_cxr=uniform_loss(cxr_on_sphere)
        # uni_loss=uni_ecg+uni_cxr

        f=torch.cat([ecg_on_sphere,cxr_on_sphere],dim=1)
        outputs = self.mlp2(f)

        return ecg_on_sphere, cxr_on_sphere, outputs



# batch_size = 4
# text_input = torch.randn(batch_size, 4096,12)         # ECG: [4, 12, 4096]
# image_input = torch.randn(batch_size, 3, 224, 224)      # CXR: [4, 3, 224, 224]

# # 实例化模型
# model = hypershpere()
# ecg_on_sphere, cxr_on_sphere, outputs, uni_loss, align_loss = model(text_input, image_input)
# print("ecg_on_sphere shape:", ecg_on_sphere.shape)
# print("cxr_on_sphere shape:", cxr_on_sphere.shape)
# print("outputs shape:", outputs.shape)
# print("uni_loss:", uni_loss.item())
# print("align_loss:", align_loss.item())