from abc import ABC
import torch
import math
from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange


class PreNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module, ABC):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

#golbal filter
from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, embed_dim, N, num_filters=1):
        super().__init__()
        self.num_filters = num_filters
        self.complex_weight = nn.Parameter(torch.randn(num_filters, N//2+1, embed_dim, 2, dtype=torch.float32) * 0.02)
        
    
    def get_cosine_factor(m, M):
        return torch.cos(
            torch.tensor(((2*m - 1) * torch.pi) / (2*M))
        ).item()

    def forward(self, x):

        # x = x.to(torch.float32)
        # x = torch.fft.rfft(x, dim=1, norm='ortho')
        power_spectrum = x**2
        all_values = []
        for filter_idx in range(self.num_filters):
            weight = torch.view_as_complex(self.complex_weight[filter_idx])
            y = x * weight * torch.cos(torch.tensor(((2*filter_idx + 1) * torch.pi) / (2 * self.num_filters)))* power_spectrum
            all_values.append(y)
        x = sum(all_values)
        x = torch.fft.irfft(x, dim=1, norm='ortho')

        return x

class Block(nn.Module):

    def __init__(self, input_size=300, embed_dim=128, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, number=196, num_filters = 2):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = GlobalFilter(embed_dim, number, num_filters)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_embed_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_embed_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(embed_dim)
        self.fremlp = FreMLP(num_tokens = number, embed_dim = embed_dim)

    def forward(self, x):
        # filtered_x = self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        filtered_x = self.drop_path(self.mlp(self.norm2(self.filter(x))))

        filtered_x = filtered_x.unsqueeze(1)
        filtered_x = self.drop_path(self.fremlp(self.norm3(filtered_x)))
        filtered_x = filtered_x.squeeze(1)
        # print(f'filtered_x {filtered_x.shape}')
        # print(f'x {x.shape}')
        # x = x + filtered_x
        # return x
        return filtered_x

class BlockECG(nn.Module):

    def __init__(self, input_size=300, embed_dim=128, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, number=4369, num_filters = 2):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = GlobalFilter_ecg(embed_dim, number, num_filters)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_embed_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_embed_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(embed_dim)
        self.fremlp = FreMLP(num_tokens = number, embed_dim = embed_dim)

    def forward(self, x):
        # filtered_x = self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        filtered_x = self.drop_path(self.mlp(self.norm2(self.filter(x))))

        filtered_x = filtered_x.unsqueeze(1)
        filtered_x = self.drop_path(self.fremlp(self.norm3(filtered_x)))
        filtered_x = filtered_x.squeeze(1)
        # print(f'filtered_x {filtered_x.shape}')
        # print(f'x {x.shape}')
        # x = x + filtered_x
        return x
        # return filtered_x

class GlobalFilter_ecg(nn.Module):
    def __init__(self, embed_dim, N, num_filters=1):
        super().__init__()
        self.num_filters = num_filters
        self.complex_weight = nn.Parameter(torch.randn(num_filters, N, embed_dim, 2, dtype=torch.float32) * 0.02)
        
    
    def get_cosine_factor(m, M):
        return torch.cos(
            torch.tensor(((2*m - 1) * torch.pi) / (2*M))
        ).item()

    def forward(self, x):

        # x = x.to(torch.float32)
        # x = torch.fft.rfft(x, dim=1, norm='ortho')
        power_spectrum = x**2
        all_values = []
        for filter_idx in range(self.num_filters):
            weight = torch.view_as_complex(self.complex_weight[filter_idx])
            y = x * weight * torch.cos(torch.tensor(((2*filter_idx + 1) * torch.pi) / (2 * self.num_filters)))* power_spectrum
            all_values.append(y)
        x = sum(all_values)
        x = torch.fft.irfft(x, dim=1, norm='ortho')

        return x

class PatchEmbed(nn.Module):
    """ fMRI to Patch Embedding
    """
    def __init__(self, num_tokens=768, kernel_size=10, stride=7):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=1, out_channels=num_tokens, kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        B, L = x.shape
        x = x.view(B, 1, L)
        x = self.proj(x).flatten(2).transpose(1,2)
        return x

class FreMLP(nn.Module):
    def __init__(self, num_tokens = 257, embed_dim = 768):
        super().__init__()
        self.embed_dim = embed_dim #embed_dim
        self.hidden_size = 1024 #hidden_size
        self.num_tokens = num_tokens
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_dim))
        self.r = nn.Parameter(self.scale * torch.randn(self.embed_dim, self.embed_dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.embed_dim, self.embed_dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.embed_dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.embed_dim))

    def forward(self, x):

        B, nd, embed_dimension, _ = x.shape
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension

        o1_real = torch.zeros([B, nd, embed_dimension // 2 + 1, self.embed_dim],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, embed_dimension // 2 + 1, self.embed_dim],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, self.r) - \
            torch.einsum('bijd,dd->bijd', x.imag, self.i) + \
            self.rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, self.r) + \
            torch.einsum('bijd,dd->bijd', x.real, self.i) + \
            self.ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
    #TODO:先注掉ifft
        # x = torch.fft.irfft(y, n=self.num_tokens, dim=2, norm="ortho")

        return x


class DFTBackbone(nn.Module):
    
    def __init__(self, input_size=5917, patch_size=450, embed_dim =512, num_tokens=[512, 256, 128, 50], depth=[2,10,2,4],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0., norm_layer=None, cls_only = False, num_filters = 1):
                 

        super().__init__()
        self.embed_dim = embed_dim
        self.cls_only = cls_only
        self.num_tokens = num_tokens  # num_features for consistency with other models
        self.num_filters = num_filters
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(num_tokens=num_tokens[0], kernel_size=patch_size, stride=patch_size)
        self.number=((input_size-patch_size)//patch_size) + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.number, num_tokens[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.tokens_down = nn.ModuleList()
        
        for i in range(len(num_tokens)-1):
            tokens_down = nn.Linear(num_tokens[i], num_tokens[i+1])
            self.tokens_down.append(tokens_down)


        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()

        cur = 0
        for i in range(len(num_tokens)):

            print('using standard block')
            blk = nn.Sequential(*[
                Block(
                input_size=input_size, embed_dim=num_tokens[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, number=self.number, num_filters = self.num_filters)
            for j in range(depth[i])
            ])

            self.blocks.append(blk)
            cur += depth[i]
    
        self.norm = norm_layer(num_tokens[-1])
        if cls_only:
            self.head = nn.Linear(num_tokens[-1], self.embed_dim)
        else:
            self.head = nn.Linear(self.number, self.embed_dim)

        self.final_dropout = nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for i in range(len(self.num_tokens)):
            x = self.blocks[i](x)
            if i != len(self.num_tokens)-1:
                x = self.tokens_down[i](x)

        x = self.norm(x)
        if self.cls_only:
            x = x.mean(1)
            x = self.final_dropout(x)
        else:
            x = self.final_dropout(x)
            x = x.transpose(1, 2)
        x = self.head(x)
        x = x.flatten(1)

        return x

#--------from MISA-----------
from torch.autograd import Function
import torch.nn as nn
import torch

"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""

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

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
#---------from MISA--------
import random
import numpy as np
seed = 42
random.seed(seed)  # 设置 Python 随机种子
np.random.seed(seed)  # 设置 NumPy 随机种子
torch.manual_seed(seed)  # 设置 PyTorch 随机种子
torch.cuda.manual_seed(seed)  # 设置当前 GPU 随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark = False


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)   #(n, d, qL)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)   #(n, cL, qL)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = torch.transpose(attn, 1, 2).contiguous() #(n, qL, cL)
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)    #(n*qL, cL)
    attn = F.softmax(attn*4, dim=-1)                #(n*qL, cL)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)   #(n, qL, cL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()    #(n, cL, qL)

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)   #(n, d, cL)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)    #(n, d, qL)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)    #(n, qL, d)

    return weightedContext, attnT


import torch
import torch.nn as nn
import torch.nn.functional as F

def activateFunc(x):
    x = torch.tanh(x)
    return F.relu(x)

class Router(nn.Module):
    def __init__(self):
        super(Router, self).__init__()
        num_out_path=32
        embed_size=32
        hid=64
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(nn.Linear(embed_size, hid), 
                                    nn.ReLU(True), 
                                    nn.Linear(hid, num_out_path))
        self.init_weights()

    def init_weights(self):
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        # print(f'x input router {x.shape}')#[bs,64,32]
        x = x.mean(-2)
        # print(f'x after mean {x.shape}')#[bs,32]
        x = self.mlp(x)
        soft_g = activateFunc(x) 
        return soft_g

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
        #随机生成一个4维数据先
        # x = torch.randn(7, 256, 7, 7)
        b, c ,_= x.size()
        # b, c = x.size()
        # y = self.avg_pool(x).view(b, c)

        # print(f'x {x.shape}')#[bs,128,4112]
        y=self.avg_pool(x)
        # print(f'after avg y {y.shape}')#[bs,128,1]
        y=y.permute(0,2,1)
        y = self.fc(y).view(b, c, 1)
        # print(f'y {y.shape}')
        # x=x.permute(0,2,1)
        return x * y.expand_as(x)
        
class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()
        embed_size=32
        self.raw_feature_norm = nn.LeakyReLU(0.1)
        self.lambda_softmax = 4
        # self.direction = direction

        self.fc_scale = nn.Linear(embed_size, embed_size)
        self.fc_shift = nn.Linear(embed_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, embed_size)
        self.fc_2 = nn.Linear(embed_size, embed_size)
        self.se=SELayer(64,8)
#new
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)

#---目前最优
#     def refine(self, query, weiContext):
#         scaling = F.tanh(self.fc_scale(weiContext))
#         shifting = self.fc_shift(weiContext)  
#         modu_res = self.fc_2(F.relu(self.fc_1(query * scaling + shifting))) 
# #----------原来版本
#         # ref_q = modu_res + query
# #----------原来版本

# #----------new版本
#         # query=self.se(query)
#         ref_q = modu_res + query


#         return ref_q
#-------目前最优

#--------改良
    def refine(self, query, weiContext):
        scaling = F.tanh(self.fc_scale(weiContext))
        # shifting = self.fc_shift(weiContext) 
        query=self.avg_pool1(query)
        #  
        modu_res = self.fc_2(F.relu(self.fc_1(query.expand_as(scaling) * scaling))) 

        ref_q = modu_res + query


        return ref_q
#--------改良版

    def forward_i2t(self, rgn, wrd):
        ref_imgs = []
        n_image = rgn.size(0)
        n_caption = wrd.size(0) 

        weiContext, attn = func_attention(rgn, wrd)
#第三版之前
        # ref_img = self.refine(rgn, weiContext)
#第三版之前

#第三版
        ref_img = self.refine(wrd, weiContext)
#第三版

        # print(f'')
        ref_img = ref_img.unsqueeze(1)
        ref_imgs.append(ref_img)

        ref_imgs = torch.cat(ref_imgs, dim=1)   #(n_img, n_stc, n_rgn, d)
        return ref_imgs       

 
    def forward(self, rgn, wrd):
        # if self.direction == 'i2t':
        ref_emb = self.forward_i2t(rgn, wrd) #(n_img, n_stc, n_rgn, d)
        # else:
        #     ref_emb = self.forward_t2i(rgn, wrd, stc_lens) #(n_img, n_stc, n_wrd, d)

        return ref_emb




#----------------------mmtm-------------------------

#----------------------mmtm----------------------------------

#------codebook------------
# import torch.nn.functional as F
# # def quantize(z):
# #     # self.num_codebook = args.num_codebook
# #     codebook = nn.Embedding(128, 64)
# #     bsz, t, dim = z.shape
# #     z = rearrange(z, 'b t d -> (b t) d')

# #     d = z.pow(2).sum(1, keepdim=True) + \
# #         self.codebook.weight.pow(2).sum(1) + \
# #         - 2 * z @ self.codebook.weight.t()

# #     min_encoding_idx = torch.argmin(d, dim=1)
# #     z_q = self.codebook(min_encoding_idx).view(z.shape)
# #     print(f'z_q {z_q.shape}')
# #     b_min_idx = rearrange(min_encoding_idx, '(b t) -> b t', t=t)
# #     print(f'z_q {z_q.shape}')
# #     encodings = torch.zeros(min_encoding_idx.shape[0], self.args.num_codebook, device=z.device)
# #     encodings.scatter_(1, min_encoding_idx.unsqueeze(1), 1)

# #     # vq loss
# #     loss_vq = F.mse_loss(z_q, z.detach())
# #     # commitment loss
# #     loss_commit = F.mse_loss(z, z_q.detach())

# #     # preserve gradients.
# #     z_q = z + (z_q - z).detach()
# #     z_q = rearrange(z_q, '(b t) d -> b t d', b=bsz)

# #     return loss_vq, loss_commit, z_q, b_min_idx

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# class VectorQuantizer(nn.Module):
#     def __init__(self):
#         super(VectorQuantizer, self).__init__()
#         self.num_codebook = 128
#         self.embedding_dim = 64
#         self.codebook = nn.Embedding(self.num_codebook, self.embedding_dim)  # Codebook (Embedding layer)
#         # self.codebook_dim = codebook_dim
#         # self.embedding_dim = embedding_dim

#     def forward(self, z):
#         bsz, t, dim = z.shape
#         z = rearrange(z, 'b t d -> (b t) d')
#         # print(f'z0 {z.shape}')

#         # Calculate distances
#         d = z.pow(2).sum(1, keepdim=True) + \
#             self.codebook.weight.pow(2).sum(1) + \
#             - 2 * z @ self.codebook.weight.t()
#         # print(f'd {d.shape}')
#         # Find closest codebook entries (indices)
#         min_encoding_idx = torch.argmin(d, dim=1)
        
#         z_q = self.codebook(min_encoding_idx).view(z.shape)
#         # print(f'z_q {z_q.shape}')

#         # Reshape indices for proper batching
#         b_min_idx = rearrange(min_encoding_idx, '(b t) -> b t', t=t)
#         # print(f'b_min_idx 0{b_min_idx.shape}')
        
#         # One-hot encoding of the min_encoding_idx
#         encodings = torch.zeros(min_encoding_idx.shape[0], self.num_codebook, device=z.device)
#         encodings.scatter_(1, min_encoding_idx.unsqueeze(1), 1)

#         # VQ loss (Vector Quantization Loss)
#         loss_vq = F.mse_loss(z_q, z.detach())
        
#         # Commitment loss
#         loss_commit = F.mse_loss(z, z_q.detach())

#         # Preserve gradients (detached z_q from the computation graph)
#         z_q = z + (z_q - z).detach()
#         z_q = rearrange(z_q, '(b t) d -> b t d', b=bsz)
#         # print(f'b_min_idx1 {b_min_idx.shape}')
#         # b_min_idx=rearrange(b_min_idx, '(b t) d -> b t d', b=bsz)
#         loss1=loss_vq+loss_commit
#         return  loss1,b_min_idx

# #------codebook--------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Collection of common utilities for distributed training. These are wrappers over
functions from :mod:`torch.distributed` module, but they do not raise exceptions
in absence of multi-GPU or CPU mode, and fall back to sensible default behavior.
"""
# from __future__ import annotations

from typing import Callable

import torch
from loguru import logger
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.distributed.nn import all_gather as nn_all_gather


def launch(
    job_fn: Callable,
    num_machines: int = 1,
    num_gpus_per_machine: int = 7,
    machine_rank: int = 0,
    dist_url: str = "tcp://ip-172-31-44-55.us-east-2.compute.internal:23456",
    args=(),
):
    """
    Launch a job in a distributed fashion: given `num_machines` machines, each
    with `num_gpus_per_machine` GPUs, this function will launch one process per
    GPU. This wrapper uses :func:`torch.multiprocessing.spawn`.

    The user has to launch one job on each machine, manually specifying a machine
    rank (incrementing integers from 0). This function will offset process ranks
    per machine. One process on `machine_rank = 0` will be the *main process*,
    and a free port on that machine will be used for process communication.

    Default arguments imply one machine with one GPU, and communication URL
    as `localhost`.

    .. note::

        We assume all machines have same number of GPUs per machine, with IDs as
        `(0, 1, 2 ...)`. If you do not wish to use all GPUs on a machine,
        set `CUDA_VISIBLE_DEVICES` environment variable appropriately.

    Args:
        job_fn: Function to launch -- this could be your model training function.
        num_machines: Number of machines, each with `num_gpus_per_machine` GPUs.
        num_gpus_per_machine: GPUs per machine, with IDs as `(0, 1, 2 ...)`.
        machine_rank: A manually specified rank of the machine, serves as a
            unique identifier and useful for assigning global ranks to processes.
        dist_url: Disributed process communication URL as `tcp://x.x.x.x:port`.
            Set this as the IP (and a free port) of machine with rank 0.
        args: Arguments to be passed to `job_fn`.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found! Cannot launch distributed processes.")

    world_size = num_machines * num_gpus_per_machine

    # Spawn `num_gpus_per_machine` processes per machine, and provide
    # "local process rank" (GPU ID) as the first arg to `_dist_worker`.
    # fmt: off
    if world_size > 1:
        mp.spawn(
            _job_worker,
            nprocs=num_gpus_per_machine,
            args=(
                job_fn, world_size, num_gpus_per_machine, machine_rank, dist_url, args
            ),
            daemon=False,
        )
    else:
        # Default to single machine, single GPU, with ID 0.
        _job_worker(0, job_fn, 1, 1, 0, dist_url, args)
    # fmt: on


def _job_worker(
    local_rank: int,
    job_fn: Callable,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: tuple,
):
    """
    Single distibuted process worker. This function should never be used directly,
    only used by :func:`launch`.
    """

    # Adjust global rank of process based on its machine rank.
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        logger.error(f"Error launching processes, dist URL: {dist_url}")
        raise e

    synchronize()
    # Set GPU ID for each process according to its rank.
    torch.cuda.set_device(local_rank)
    job_fn(*args)


def synchronize() -> None:
    """Synchronize (barrier) all processes in a process group."""
    if dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    """Return number of processes in the process group, each uses 1 GPU."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Return rank of current process in the process group."""
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    """
    Check whether current process is the main process. This check is useful
    to restrict logging and checkpointing to main process. It will always
    return `True` for single machine, single GPU execution.
    """
    return get_rank() == 0


# def gather_across_processes(t: torch.Tensor) -> list[torch.Tensor]:
#     """
#     Gather tensors from multiple GPU processes in a list. The order of elements
#     is preserved by GPU process IDs. This operation is differentiable; gradients
#     will be scattered back to devices in the backward pass.

#     Args:
#         t: Tensor to gather across processes.
#     """
#     world_size = dist.get_world_size()
#     if world_size == 1:
#         return [t]

#     output = list(nn_all_gather(t))
    
#     return output

def gather_across_processes(t: torch.Tensor) -> list[torch.Tensor]:
    """
    On a single GPU (world_size==1), just return [t].
    Otherwise do an all_gather.
    """
    if not dist.is_available():
        return [t]

    world_size = dist.get_world_size()
    if world_size <= 1:
        # No need for any rendezvous
        return [t]

    # multi-GPU / multi-process case
    output = list(nn_all_gather(t))
    return output


def gpu_mem_usage() -> int:
    """
    Return gpu memory usage (in megabytes). If not using GPU, return 0 without
    raising any exceptions.
    """
    if torch.cuda.is_available():
        # This will be in bytes, so we divide by (1024 * 1024).
        return torch.cuda.max_memory_allocated() // 1048576
    else:
        return 0




#-------------
import itertools

import torch
import torch.nn.functional as F


def zeroshot_retrieval_logits(r_x, rep_list, logit_scale_exp, loss_fn):
    """
    Computes logits for zeroshot retrieval based on the specified loss function.

    Calculates the logits for predicting the modality r_x using the representations
    in rep_list, and scales the logits by the exponentiated logit scale parameter.

    Args:
        r_x (torch.Tensor): Encoded representations of the modality to predict (num_candidates, d).
        rep_list (list[torch.Tensor]): List of representations for the remaining modalities, each of
                                       size (batch_sz, d) or (d,). This list can can be of any length.
        logit_scale_exp (torch.Tensor): Exponentiated logit scale parameter.
        loss_fn (str): The loss function to use, either "symile" or "clip".

    Returns:
        Tensor: Logits for zeroshot retrieval, of shape (batch_sz, num_candidates).
    """
    if loss_fn == "symile":
        # logits is a (batch_sz, n) matrix where each row i is
        # [ MIP(r_x[i], r_y[i], r_z[0]) ... MIP(r_x[i], r_y[i], r_z[n-1]) ]
        # where MIP is the multilinear inner product.
        product = torch.ones_like(rep_list[0])
        for r in rep_list:
            product *= r

        logits = product @ torch.t(r_x)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
    elif loss_fn == "clip":
        # logits is a (batch_sz, n) matrix where each row i is
        # [ r_x[i]^T r_z[0] + r_z[0]^T r_y[i]   + r_x[i]^T r_y[i] ...
        #   r_x[i]^T r_z[n-1] + r_z[n-1]^T r_y[i] + r_x[i]^T r_y[i] ]
        for i in range(len(rep_list)):
            rep_list[i] = rep_list[i].unsqueeze(0) if rep_list[i].dim() == 1 else rep_list[i] # (batch_sz, d)

        pairwise_sum_with_r_x = torch.zeros_like(rep_list[0] @ torch.t(r_x)) # (batch_sz, num_candidates)
        for r in rep_list:
            pairwise_sum_with_r_x += r @ torch.t(r_x)

        pairwise_sum_without_r_x = torch.zeros((rep_list[0].shape[0], 1), device=rep_list[0].device) # (batch_sz, 1)
        for x, y in itertools.combinations(rep_list, 2):
            pairwise_sum_without_r_x += torch.diagonal(x @ torch.t(y)).unsqueeze(dim=1)

        logits = pairwise_sum_with_r_x + pairwise_sum_without_r_x

    assert logits.dim() == 2, "Logits must be a 2D tensor."

    return logit_scale_exp * logits


########
# clip #
########


def infonce(u, v, logit_scale):
    """
    Computes the CLIP (InfoNCE) loss for a batch of representations.

    Args:
        u, v (torch.Tensor): Representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): Learned temperature parameter.
    Returns:
        (torch.Tensor): CLIP (InfoNCE) loss
    """
    logits_u = logit_scale * u @ v.T
    logits_v = logit_scale * v @ u.T

    assert logits_u.shape == logits_v.shape, "Joint embedding spaces must be the same shape."
    labels = torch.arange(logits_u.shape[0]).to(u.device)
    return (F.cross_entropy(logits_u, labels) + F.cross_entropy(logits_v, labels)) / 2.0

def clip(r_a, r_b, r_c, logit_scale, negative_sampling=None):
    """
    Computes the pairwise CLIP loss for a batch of representations.

    Args:
        r_a, r_b, r_c (torch.Tensor): Representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (None): Argument is included for compatibility but is not used in the function.
    Returns:
        (torch.Tensor): Average over the pairwise CLIP (InfoNCE) losses
    """
    loss_ab = infonce(r_a, r_b, logit_scale)
    loss_bc = infonce(r_b, r_c, logit_scale)
    loss_ac = infonce(r_a, r_c, logit_scale)
    return loss_ab + loss_bc + loss_ac


##########
# symile #
##########


def compute_logits_neg_sampling_n(x, y, z):
    """
    Computes the logits for anchor modality x with batch_sz - 1 negatives for
    each positive - or (batch_sz^2 - batch_sz) total negatives.

    If batch_sz is n, then returned logits have size (n, n) with n positive
    multilinear inner products and (n^2 - n) negative multilinear inner products.

    Positive multilinear inner products (MIPs) are along the diagonal of the
    square logits matrix. For example, the second row of `logits` might be:

    [ MIP(x[1], y[3], z[2]) MIP(x[1], y[1], z[1]) MIP(x[1], y[0], z[1]) MIP(x[1], y[2], z[3]) ].

    Notice that only the second element is the positive MIP; all others are negative.
    There is a small chance of a false negative MIP.

    Args:
        x (torch.Tensor): Representation vector of size (batch_sz, d_r).
        y (torch.Tensor): Representation vector of size (batch_sz, d_r).
        z (torch.Tensor): Representation vector of size (batch_sz, d_r).
    Returns:
        logits (torch.Tensor): Logits for x of size (batch_sz, batch_sz).
    """
    # shuffle rows of y and z
    y_shuff = y[torch.randperm(y.shape[0])]
    z_shuff = z[torch.randperm(z.shape[0])]
    logits_x = x @ torch.t(y_shuff * z_shuff) # (batch_sz, batch_sz)
    MIP_of_pos_triples = (x * y * z).sum(axis=1) # (batch_sz)
    # insert positive triples along diagonal of shuffled logits
    return torch.where(torch.eye(n=x.shape[0]).to(x.device) > 0.5, MIP_of_pos_triples, logits_x)


def compute_logits_neg_sampling_n_squared(x, y, z):
    """
    Computes the logits for anchor modality x with batch_sz^2 - 1 negatives for
    each positive.

    If batch size is n, then returned logits have size (n, n^2) with n positive
    multilinear inner products and (n^3 - n) negative multilinear inner products.

    Positive multilinear inner products (MIP) are along the main diagonal of the
    (non-square) logits matrix. For example, if n = 4, then the second row of
    `logits` is:

    [ MIP(x[1], y[0], z[0]) MIP(x[1], y[1], z[1]) MIP(x[1], y[2], z[2]) MIP(x[1], y[3], z[3])
      MIP(x[1], y[0], z[3]) MIP(x[1], y[1], z[0]) MIP(x[1], y[2], z[1]) MIP(x[1], y[3], z[2])
      MIP(x[1], y[0], z[2]) MIP(x[1], y[1], z[3]) MIP(x[1], y[2], z[0]) MIP(x[1], y[3], z[1])
      MIP(x[1], y[0], z[1]) MIP(x[1], y[1], z[2]) MIP(x[1], y[2], z[3]) MIP(x[1], y[3], z[0])  ]

    Notice that only the second element is the positive MIP; all others are negative.

    Args:
        x (torch.Tensor): Representation vector of size (batch_sz, d_r).
        y (torch.Tensor): Representation vector of size (batch_sz, d_r).
        z (torch.Tensor): Representation vector of size (batch_sz, d_r).
    Returns:
        logits (torch.Tensor): Logits for x of size (batch_sz, batch_sz^2).
    """
    y_z = []
    for i in range(y.shape[0]):
        y_z.append(y * z)
        z = torch.roll(z, shifts=1, dims=0)

    # concatenate elements in y_z so that y_z has shape (n^2, d) where each row
    # is a different element-wise product of a row from y and a row from z
    y_z = torch.cat(y_z, 0)

    # return logits with shape (n, n^2) where each row is the multilinear inner
    # product between that row in x and each row from y_z
    logits = x @ y_z.T
    return logits


def symile(r_a, r_b, r_c, logit_scale, negative_sampling):
    """
    Computes the Symile loss for a batch of representations. The final Symile
    loss is an average of the loss terms where each modality is treated as the
    anchor in turn.

    The argument `negative_sampling` can take on one of two values:
        - `n` (for O(n)): draws n - 1 negative samples for each positive
        - `n_squared` (for O(n^2)): draws n^2 - 1 negative samples for each positive

    Args:
        r_a, r_b, r_c (torch.Tensor): Representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (str): Specifies the negative sampling strategy.
                                 Must be either `n` or `n_squared`.

    Returns:
        (torch.Tensor): Average over the losses where each modality is treated
                        as the anchor in turn.
    """
    if negative_sampling == "n":
        logits_a = logit_scale * compute_logits_neg_sampling_n(r_a, r_b, r_c)
        logits_b = logit_scale * compute_logits_neg_sampling_n(r_b, r_a, r_c)
        logits_c = logit_scale * compute_logits_neg_sampling_n(r_c, r_a, r_b)
    elif negative_sampling == "n_squared":
        logits_a = logit_scale * compute_logits_neg_sampling_n_squared(r_a, r_b, r_c)
        logits_b = logit_scale * compute_logits_neg_sampling_n_squared(r_b, r_a, r_c)
        logits_c = logit_scale * compute_logits_neg_sampling_n_squared(r_c, r_a, r_b)
    else:
        raise ValueError("negative_sampling must be either 'n' or 'n_squared'.")

    labels = torch.arange(logits_a.shape[0]).to(r_a.device)
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0