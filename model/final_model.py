。import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from scipy.signal import stft, istft
import torchaudio
import sys
import os
sys.path.append(os.path.abspath('/home/ke/MIMIC_subset/MIMIC_subset'))
from model.fusion_model import FtBlock
from model.share_spec import IMFM,DiffLoss, MSE, SIMSE,ImagePatchEmbed,FeedForward,AddNorm
# import model.configs as configs
from model.utils import func_attention,Router,Refinement,gather_across_processes,get_rank
from model.xlstm_used import xLSTM
import torch.nn as nn
import torchvision
import torch
import numpy as np
from model.graph_att import GraphAttentionLayerFusion,CrossAttentionLayer
from model.hpblic_transform import hyperCA
from torch.nn.functional import kl_div, softmax, log_softmax
import model.lorentz as L
# from .loss import RankingLoss, CosineLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()



import torch.nn as nn
import numpy as np

#========senet===============
from torch import nn
import random
seed = 42
random.seed(seed)  
np.random.seed(seed) 
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
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

        if n_filters_in != n_filters_out:

            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]

        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""

        if self.skip_connection is not None:
            # print(f'start skip_connection')
            y = self.skip_connection(y)
        else:
            y = y

        x = self.conv1(x)
        # print(f'after blk first conv1 {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

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
        self.blocks_dim=list(zip(args.mod_net_filter_size, args.mod_net_seq_lengh))

        n_filters_in, n_filters_out = 12, self.blocks_dim[0][0]
        n_samples_in, n_samples_out = 4096, self.blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.res_blocks = nn.ModuleList()
        for i, (n_filters, n_samples) in enumerate(self.blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            

            resblk1d = ResBlock1d( n_filters_in,n_filters_out, downsample, kernel_size, dropout_rate)

            self.res_blocks.append(resblk1d)

        n_filters_last, n_samples_last = self.blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, 512)

        self.lin1 = nn.Linear(512, 128)
        self.n_blk = len(self.blocks_dim)
        self.lin128 = nn.Linear(320, 128)

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        y = x
        for i,blk in enumerate(self.res_blocks):

            x, y = blk(x, y)

        x=x.permute(0,2,1)
        x=self.lin128(x)

        return x


class ResBlock_fre(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out,kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock_fre, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        n_filters_in=128
        n_filters_out=128
        kernel_size=7
        padding=3
        stride=2
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=3,stride=4, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        # padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=4, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Residual unit."""
        x = self.conv1(x)
        # print(f'after blk first conv1 {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        # x_after_dropout1 = self.dropout1(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.relu(x)

        return x

#================resnet1d===================








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


class ResBlock_frecxr(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out,kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock_frecxr, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        n_filters_in=128
        n_filters_out=128
        kernel_size=5

        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=2,stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        # padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)



    def forward(self, x):
        """Residual unit."""

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)

        x = self.bn2(x)
        x = self.relu(x)

        return x


#==========resnet34=============





import torch
import torch.nn as nn
import math
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

        self.pool = nn.AdaptiveMaxPool1d(output_size=16)

    def forward(self, cxr):
        # print(f'cxr input {cxr.shape}')
        cxr = self.conv1(cxr)  # [bs, 64, 25]
        cxr = self.pool(cxr)   # [bs, 64, 32]
        # cxr=nn.LayerNorm(cxr)
        return cxr

class final_extract(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        # self.layernorm=nn.LayerNorm(d_model)

    def forward(self, cxr):
        cxr = self.conv1(cxr)  # [bs, 64, 25]
        cxr = self.pool(cxr)   # [bs, 64, 32]
        # cxr=nn.LayerNorm(cxr)
        return cxr



class final_fusion(nn.Module):
    def __init__(self, channel, reduction=4):
        super(final_fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x,y):
        b, c ,_= x.size()
        x=self.avg_pool(x)
        # print(f'after avg y {y.shape}')#[bs,128,1]
        x=x.permute(0,2,1)
        x = self.fc(x).view(b, c, 1)
        # print(f'y {y.shape}')
        # x=x.permute(0,2,1)
        return y * x.expand_as(y)+y


class final_Res1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self):
        super(final_Res1d, self).__init__()
        # Forward path
        # padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1,stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.3)
        # padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1,stride=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=8)


    def forward(self, x):

        x = self.conv1(x)
        # print(f'after blk first conv1 {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x=self.maxpool(x)
        # print(f'after maxpool x {x.shape}')
        x = self.relu(x)
        # x = self.dropout2(x)
        return x


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum()



class Fusion(nn.Module):
    def __init__(self, d_model, act_layer=torch.tanh):
        super(Fusion, self).__init__()

        self.text_weight = nn.Parameter(torch.randn(128, 128, dtype=torch.float32))
        self.image_weight = nn.Parameter(torch.randn(128, 128, dtype=torch.float32))
        self.fusion_weight = nn.Parameter(torch.randn(128, 128, dtype=torch.float32))
        self.act_layer = act_layer
        self.a = nn.Parameter(torch.tensor(0.5))
        self.ecg_extractor = ECGFeatureExtractor()
        self.cxr_extractor = CXRFeatureExtractor()

        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.5))
        self.se=SELayer(64,8)

        self.final_fusion1=final_fusion(64,8)
        self.final_fusion2=final_fusion(64,8)
        self.final1=final_Res1d()
        self.final2=final_Res1d()

        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.a3 = nn.Parameter(torch.tensor(1.0))
        self.c=nn.Parameter(torch.tensor(1.0))
        self.d=nn.Parameter(torch.tensor(1.0))

        embed_dim=64
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.cli_alpha = nn.Parameter(torch.tensor(9**-0.5).log())
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(1.0).log(), requires_grad=True
        )

        self._curv_minmax = {
            "max": math.log(1.0 * 10),
            "min": math.log(1.0 / 10),
        }
        self._rank = get_rank()
        self.mlp1 = MLP(64, int(64*2), 64, 9, dropout=0.5)
        self.mlp2 = MLP(64, int(64*2), 64, 9, dropout=0.5)
        self.delta_image = nn.Parameter(torch.tensor(0.1))  
        self.delta_text = nn.Parameter(torch.tensor(0.1))   
        self.lambda_reg = 0.01  
        self.hyperCA=hyperCA(128,32)

 
    

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def forward(self, text, image,cli):
        text = self.ecg_extractor(text)  # [bs, 64, 32]
        image = self.cxr_extractor(image)
        bs=text.size(0)

        text = torch.max(text, dim=2)[0]
        image = torch.max(image, dim=2)[0]

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)
        self.cli_alpha.data = torch.clamp(self.cli_alpha.data, max=0.0)

        image_feats = image * self.visual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):

            image_feats = L.exp_map0(image_feats, self.curv.exp())


        text_feats = text * self.textual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            text_feats = L.exp_map0(text_feats, self.curv.exp())
            # print(f'text_feats before cat {text_feats}')

        cli_feats = cli * self.cli_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            cli_feats = L.exp_map0(cli_feats, self.curv.exp())
            # print(f'text_feats before cat {text_feats}')

        concate=torch.cat([image_feats,text_feats],dim=1)#[bs,128]
        # print(f'concate {concate.shape}')
        f_hptrans=self.hyperCA(concate)
        all_image_feats = image_feats
        all_text_feats = text_feats
        with torch.autocast(self.device.type, dtype=torch.float32):
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)

        batch_size = image_feats.shape[0]
        targets = torch.arange(batch_size, device=image_logits.device)
        targets = targets + batch_size * self._rank

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        contrastive_loss = 0.5 * (
            nn.functional.cross_entropy(_scale * image_logits, targets)
            + nn.functional.cross_entropy(_scale * text_logits, targets)
        )

        image_feats_pool = image_feats.mean(1).unsqueeze(1)

        text_feats_pool = text_feats.mean(1).unsqueeze(1)

        f1=torch.cat([image_feats,text_feats,cli_feats],dim=1)

        image_feats=self.mlp1(image_feats)

        _angle1 = L.oxy_angle(image_feats, cli_feats, _curv)
        # print(f'_angle1 {_angle1}')
        _aperture1 = L.half_aperture(image_feats, _curv)
        # print(f'_aperture1 {_aperture1}')

        entailment_loss1 = torch.clamp(_angle1 - _aperture1- self.delta_image, min=0).mean()+self.lambda_reg * self.delta_image**2

        text_feats=self.mlp2(text_feats)
        _angle2 = L.oxy_angle(text_feats, cli_feats, _curv)
        _aperture2 = L.half_aperture(text_feats, _curv)
        entailment_loss2 = torch.clamp(_angle2 - _aperture2- self.delta_text, min=0).mean()+self.lambda_reg * self.delta_text**2

        loss = contrastive_loss
        total_loss = loss + 0.2 * entailment_loss1+ 0.2 * entailment_loss2
        # total_loss = loss
        # print(f'loss {loss}')
        f=torch.cat([image_feats,text_feats,cli_feats],dim=1)
        print(f'image_feats {image_feats.shape}')
        print(f'text_feats {text_feats.shape}')
        print(f'cli_feats {cli_feats.shape}')

        return f,f1,f_hptrans,total_loss








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

class domain_fusion(nn.Module):
    def __init__(self, in_channels=128, reduction_factor=8):
        super(domain_fusion, self).__init__()

        self.intermediate_channels = in_channels // reduction_factor

        self.local_attention_layers = nn.ModuleList([
            nn.Conv1d(in_channels, self.intermediate_channels, kernel_size=1),
            nn.BatchNorm1d(self.intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.intermediate_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
        ])

        self.global_attention_layers = nn.ModuleList([
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(in_channels, self.intermediate_channels, kernel_size=1),
            nn.BatchNorm1d(self.intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.intermediate_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
        ])

        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, x, y):
        combined_input = x + y

        local_attention = combined_input
        for layer in self.local_attention_layers:
            local_attention = layer(local_attention)
        # print(f'combined_input {combined_input.shape}')
        global_attention = self.global_attention_layers[0](combined_input)
        for layer in self.global_attention_layers[1:]:
            global_attention = layer(global_attention)

        attention_map = self.sigmoid_activation(local_attention + global_attention)

        output = 2 * x * attention_map + 2 * y * (1 - attention_map)
        return output
        
           
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat((x, out), 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DenseNet1D(nn.Module):
    def __init__(self, input_dim, growth_rate, num_layers, num_classes):
        super(DenseNet1D, self).__init__()
        self.conv0 = nn.Conv1d(input_dim, 2 * growth_rate, kernel_size=3, padding=1)
        self.relu0 = nn.ReLU(inplace=True)
        self.dense_block = DenseBlock(2 * growth_rate, num_layers, growth_rate)
        self.fc = nn.Linear(2 * growth_rate + num_layers * growth_rate, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.dense_block(x)
        x = x.mean(dim=2)  # 全局平均池化
        x = self.fc(x)
        return x

class DDMF_Net_hpblic7_2(nn.Module):
    def __init__(self,  d_text=12, seq_len=4369, img_size=224, patch_size=16, d_model=128,
                 num_filter=2, num_class=3, num_layer=1, dropout=0., mlp_ratio=4.):
        super(DDMF_Net_hpblic7_2, self).__init__()

        # Text

        self.text_encoder = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=d_model, kernel_size=17, stride=1, padding=8),  # 卷积层

       )

        self.ecg_norm=nn.LayerNorm(d_model)
        s=seq_len


        # Image
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        self.fusion = Fusion(d_model)

        self.mlp = MLP(d_model, int(mlp_ratio*d_model), d_model, num_class, dropout=dropout)
        self.mlp1 = MLP(64, int(mlp_ratio*64), 64, num_class, dropout=0.3)
        self.mlp2 = MLP(27, int(2*27), 27, num_class, dropout=0.3)
        self.mlp4 = MLP(137, int(2*137), 137, num_class, dropout=0.3)
        self.mlp5 = MLP(128, int(mlp_ratio*128), 128, num_class, dropout=0.3)

        # self.mlp3 = MLP(9, int(mlp_ratio*9), 9, 9, dropout=0.3)
        # self.mlp1=Classifier(3)
        # self.mlp3 = DenseNet1D(9, 32, 4, 9)
        # self.mlp4 = MLP(137, int(2*137), 64, num_class, dropout=0.3)
        self.mlp3 = DenseNet1D(9, 32, 4, 9)
#
        # DenseNet1D(input_dim=9, growth_rate=32, num_layers=4, num_classes=64)


        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

        self.resnet1d=ResNet1d()
        self.layernorm = nn.LayerNorm(128)
        self.act_layer = nn.ReLU()
        self.cxrmodel=CXRModels()
        self.se=SELayer(128,16)
        self.cxrlin=nn.Linear(512, 128)
        self.blk_fre=ResBlock_fre(128,128,5,0.5)
        self.blk_fre_cxr=ResBlock_frecxr(128,128,5,0.5)

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
        self.cxr_fusion=domain_fusion()
        self.ecg_fusion=domain_fusion()
        # self.vitlin=nn.Linear(768, 128)
        # self.vitlin2=nn.Conv1d(197, 49, kernel_size=1)
        self.xLSTMLMModel= xLSTM(input_size=4096, head_size=1024, num_heads=2, batch_first=True, layers='ms')



    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            # nn.init.constant_(m.bias.data, 0.0)
            # trunc_normal_(m.weight, std=.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image,clic_value):
        # print(f'start hpblic7_1--------')
        
        
        text=text.permute(0,2,1)
        ecg_temporal=self.resnet1d(text)
        ecg_temporal=ecg_temporal.permute(0,2,1)
        text = self.text_encoder(text)
        # print(f'after encoder ecg {text.shape}')
        text=text.permute(0,2,1)
        text=self.ecg_norm(text)

        image = image.to(torch.float32)
        cxr_spatial=self.cxrmodel(image)
        cxr_spatial=self.conv_cxr1(cxr_spatial)
        bs,c,h,w=cxr_spatial.shape
        cxr_spatial=cxr_spatial.view(bs,c,h*w)
#-------resnet34----------

        image = self.img_patch_embed(image)
        # print(f'after iamge embedding {image.shape}')
        image = image + self.img_pos_embed
        image = self.img_pos_drop(image)

        text, image = self.FourierTransormer(text, image)
        image=image.permute(0,2,1)
        image=self.blk_fre_cxr(image)

        text=text.permute(0,2,1)
        text=self.blk_fre(text)
 
        image=self.act_layer(image)
        text=self.act_layer(text)
        ecg_temporal=self.act_layer(ecg_temporal)
        text=self.ecg_fusion(text,ecg_temporal)
        # text=self.act_layer(text)
        cxr_spatial=self.act_layer(cxr_spatial)
        text=self.se(text)

        image=self.cxr_fusion(image,cxr_spatial)
        image=self.se(image)
        clic_value = clic_value.unsqueeze(2)
        
        clic_value=self.mlp3(clic_value)
        # print(f'clic_value{clic_value.shape}')
        # tensor.squeeze(-1)
        clic_value = clic_value.squeeze(dim=-1)



        f,f1,concate,loss1 = self.fusion(text, image,clic_value)  # (batch, d_model)



        outputs1 = self.mlp5(concate)
        outputs2 = self.mlp2(f)
        outputs3 = self.mlp4(f1)
        # outputs_e=self.mlp4(fused)

        outputs=0.25*outputs1+0.25*outputs2+0.5*outputs3

        return text, image, outputs, loss1



