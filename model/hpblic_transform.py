import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.init as init
# from geoopt import ManifoldParameter
# from geoopt.optim.rsgd import RiemannianSGD
# from geoopt.optim.radam import RiemannianAdam
import math
from typing import Tuple, Optional
# from geoopt import Lorentz as LorentzOri

# class Lorentz():
#     def __init__(self, k=1.0, learnable=False):
#         """
#         Initialize a Lorentz manifold with curvature k.

#         Parameters:
#             k (float): Curvature of the manifold.
#             learnable (bool): If True, k is learnable. Default is False.
#         """
#         super().__init__(k, learnable)

# class Lorentz:
#     def __init__(self, k: float = 1.0, learnable: bool = False):
#         """
#         Initialize a Lorentz manifold with curvature k.

#         Parameters:
#             k (float): Curvature of the manifold. Default is 1.0.
#             learnable (bool): If True, k is learnable. Default is False.

#         Example:
#             lorentz_manifold = Lorentz(k=0.5, learnable=True)
#         """
#         super().__init__(k, learnable)  # 确保正确的父类

class Lorentz:
    def __init__(self, k: float = 1.0, learnable: bool = False):
        """
        Initialize a Lorentz manifold with curvature k.

        Parameters:
            k (float): Curvature of the manifold. Default is 1.0.
            learnable (bool): If True, k is learnable. Default is False.
        """
        self.k = k
        self.learnable = learnable


def expmap00(u, *, k, dim=-1):
    r"""
    Compute exponential map for Hyperboloid from :math:`0`.

    Parameters
    ----------
    u : tensor
        speed vector on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)


def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        # return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
        #     dim=dim, keepdim=True
        # )
        return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(
            dim=dim, keepdim=True
        )
from typing import Tuple, Any, Union, List
eps=1e-6
class LeakyClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        with torch.no_grad():
            ctx.save_for_backward(x.ge(min) & x.le(max))
            return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None
def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)
def sqrt(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=1e-9)  # Smaller epsilon due to precision around x=0.
    return torch.sqrt(x)
def _norm(u, keepdim: bool = False, dim: int = -1):
    return sqrt(_inner(u, u, keepdim=keepdim))
EXP_MAX_NORM = 10.
def _expmap0(u, k: torch.Tensor, dim: int = -1):
    # nomin = (_norm(u, keepdim=True, dim=dim) / torch.sqrt(k)).clamp_max(10.)
    nomin = (_norm(u, keepdim=True, dim=dim))
    u = u / nomin
    nomin = nomin.clamp_max(EXP_MAX_NORM)
    # mask = nomin.lt(EXP_MAX_NORM)
    # if (~mask).any():
    #     nomin_mask = nomin.masked_scatter(mask, torch.ones_like(nomin))
    #     u = u / nomin_mask
    #     nomin = (_norm(u, keepdim=True, dim=dim))
    l_v = torch.cosh(nomin)
    r_v = torch.sinh(nomin) * u
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p

def expmap0( u: torch.Tensor, *, project1=True, dim=-1) -> torch.Tensor:
    """
    Perform the exponential map from the origin.

    Parameters:
        u (torch.Tensor): Tangent vector.
        project (bool): If True, project the result back onto the manifold. Default is True.
        dim (int): Dimension to perform the operation.

    Returns:
        torch.Tensor: Point on the manifold.
    """

    res = expmap00(u, k=torch.tensor(1.0), dim=dim)
    if project1:
        return project(res, k=torch.tensor(1.0), dim=dim)
    else:
        return res


def project(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{k+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim)


@torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    right_ = x.narrow(dim, 1, dn)
    left_ = torch.sqrt(
        k + (right_ * right_).sum(dim=dim, keepdim=True)
    )
    x = torch.cat((left_, right_), dim=dim)
    return x

class HypLinear(nn.Module):
    """
    Hyperbolic Linear Layer

    Parameters:
        manifold (Manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.0.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, dropout=0.0, manifold_out=None):
        super().__init__()
        self.in_features = in_features + 1  # +1 for time dimension
        self.out_features = out_features
        self.bias = bias
        self.manifold = Lorentz(k=float(1.0))
        self.manifold_out = Lorentz(k=float(1.0))

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.dropout_rate = dropout
        self.reset_parameters()
        # self.expmap0=expmap0()

    def reset_parameters(self):
        """Reset layer parameters."""
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp'):
        """Forward pass for hyperbolic linear layer."""
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            print(f'hplinear x1 {x}')
            x = expmap0(x)
            print(f'hplinear x2 {x}')
        x_space = self.linear(x)
        print(f'hplinear x_space {x_space}')

        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + torch.tensor(1.0)).sqrt()
        print(f'hplinear x_time {x_time}')
        x = torch.cat([x_time, x_space], dim=-1)
        print(f'hplinear x3 {x}')
        if self.manifold_out is not None:
            x = x .sqrt()
            print(f'hplinear x3 {x}')
        return x

class HypLayerNorm(nn.Module):
    """
    Hyperbolic Layer Normalization Layer

    Parameters:
        manifold (Manifold): The manifold to use for normalization.
        in_features (int): The number of input features.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, in_features, manifold_out=None):
        super(HypLayerNorm, self).__init__()
        self.in_features = in_features
        self.manifold = Lorentz(k=float(1.0))
        self.manifold_out = Lorentz(k=float(1.0))
        self.layer = nn.LayerNorm(128)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.layer.reset_parameters()

    def forward(self, x):
        """Forward pass for hyperbolic layer normalization."""
        # x_space = x[..., 1:]
        x_space = self.layer(x)
        # x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + torch.tensor(1.0)).sqrt()
        # x = torch.cat([x_time, x_space], dim=-1)

        if self.manifold_out is not None:
            x = x .sqrt()
        return x



class HypActivation(nn.Module):
    """
    Hyperbolic Activation Layer

    Parameters:
        manifold (Manifold): The manifold to use for the activation.
        activation (function): The activation function.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, activation, manifold_out=None):
        super(HypActivation, self).__init__()
        self.manifold = Lorentz(k=float(1.0))
        self.manifold_out = Lorentz(k=float(1.0))
        self.activation = activation

    def forward(self, x):
        """Forward pass for hyperbolic activation."""
        x_space = x[..., 1:]
        x_space = self.activation(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) +torch.tensor(1.0)).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k /torch.tensor(1.0)).sqrt()
        return x


class HypDropout(nn.Module):
    """
    Hyperbolic Dropout Layer

    Parameters:
        manifold (Manifold): The manifold to use for the dropout.
        dropout (float): The dropout probability.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, dropout, manifold_out=None):
        super(HypDropout, self).__init__()
        self.manifold = Lorentz(k=float(1.0))
        self.manifold_out = Lorentz(k=float(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=False):
        """Forward pass for hyperbolic dropout."""
        if training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) +torch.tensor(1.0)).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.k /torch.tensor(1.0)).sqrt()
        return x



def cinner( x, y):
    x = x.clone()
    x.narrow(-1, 0, 1).mul_(-1)
    return x @ y.transpose(-1, -2)

def inner( x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, *, keepdim=False,
              dim=-1) -> torch.Tensor:

    if v is None:
        v = u
    return _inner(u, v, dim=dim, keepdim=keepdim)

              
    # if v is None:
    #     v=u
        # v = u
    
def cinner( x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    x = x.clone()
    x.narrow(-1, 0, 1).mul_(-1)
    return x @ y.transpose(-1, -2)

def mid_point( x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:

    if w is not None:
        ave = w.matmul(x)
    else:
        ave = x.mean(dim=-2)
    denom = (-inner(ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt()
    return torch.tensor(1.0).sqrt() * ave / denom



class hyperCALayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = Lorentz(k=float(1.0))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        # self.attention_type = args.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        # self.power_k = args.power_k
        # self.trans_heads_concat = args.trans_heads_concat


    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def full_attention(self, qs, ks, vs, output_attn=False):
        # normalize input
        # qs = HypNormalization(self.manifold)(qs)
        # ks = HypNormalization(self.manifold)(ks)

        # negative squared distance (less than 0)
        att_weight = 2 + 2 * cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H, N, N]
        att_weight = att_weight / self.scale + self.bias  # [H, N, N]

        att_weight = nn.Softmax(dim=-1)(att_weight)  # [H, N, N]
        att_output = mid_point(vs.transpose(0, 1), att_weight)  # [N, H, D]
        att_output = att_output.transpose(0, 1)  # [N, H, D]

        att_output = mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(query_input)
            k_list.append(source_input)
            if self.use_weight:
                v_list.append(source_input)
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)  # [N, H, D]
        key = torch.stack(k_list, dim=1)  # [N, H, D]
        value = torch.stack(v_list, dim=1)  # [N, H, D]

        # if output_attn:

        attention_output = self.full_attention(
            query, key, value, output_attn)
       

        final_output = attention_output
        # multi-head attention aggregation
        # final_output = self.manifold.mid_point(final_output)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class hyperCA(nn.Module):
    def __init__(self,in_channels, hidden_channels,  manifold_in=Lorentz(k=float(1.0)), manifold_hidden=Lorentz(k=float(1.0)), manifold_out=Lorentz(k=float(1.0)), num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # self.out_channels = out_channels
        self.num_layers = 2
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        # self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden)
        # self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                hyperCALayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels, num_heads=self.num_heads, use_weight=self.use_weight, args=args))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

        # self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)

    def forward(self, x):
        layer_ = []

        # x = self.fcs[0](x_input, x_manifold='euc')
        # print(f'x1 {x}')

                
        # x = self.bns[0](x)
        # # print(f'x2 {x}')

        # x = self.activation(x)
        # # print(f'x3 {x}')
        # x = self.dropout(x, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            # print(f'x4 {x}')
            # if self.residual:
            #     x = mid_point(torch.stack((x, layer_[i]), dim=1))
            #     # print(f'x5 {x}')
            # if self.use_bn:
            #     x = self.bns[i + 1](x)
                # print(f'x6 {x}')
            # if self.use_act:
            #     x = self.activation(x)
            # # x = self.dropout(x, training=self.training)
            layer_.append(x)

        # x = self.fcs[-1](x)
        
        # print(f'x7 {x}')
        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]
