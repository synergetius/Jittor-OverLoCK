import jittor
from jittor import Module
from jittor import nn
from jittor.einops import rearrange
from jittor.linalg import einsum
import numpy as np
import os
import hashlib
def get_conv2d(in_channels, 
               out_channels, 
               kernel_size, 
               stride, 
               padding, 
               dilation, 
               groups, 
               bias,
               attempt_use_lk_impl=True):
    
    kernel_size = (kernel_size, kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = (padding, padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
    ########## 这里关于大卷积核的高性能实现需要安装或自己重新实现。研究一下
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding, 
                     dilation=dilation, 
                     groups=groups, 
                     bias=bias)
def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)
def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim//2),
        nn.GELU(),
        nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim//2),
        nn.GELU(),
        nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim)
    )
def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )       
class SEModule(Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )
        
    def execute(self, x):
        x = x * self.proj(x)
        return x
class LayerScale(Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = jittor.ones(dim, 1, 1, 1)*init_value
        self.bias = jittor.zeros(dim)

    def execute(self, x):
        x = nn.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    def execute(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().execute(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    # ConvFFN中的归一化模块，可以了解一下具体的原理
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = jittor.zeros(1, dim, 1, 1)
        if self.use_bias:
            self.beta = jittor.zeros(1, dim, 1, 1)

    def execute(self, x):
        Gx = jittor.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        # 对空间维度求2-范数，再对通道求平均后，作为分母执行归一化
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x
    

class DilatedReparamBlock(Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def execute(self, x):
        if not hasattr(self, 'origin_bn'): # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        print("merge")
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))
class CTXDownsample(Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim)
        )
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim//4, h_dim//4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim//4)
        )

    def execute(self, x, ctx):
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return (x, ctx)
class ResDWConv(nn.Conv2d):
    '''
    Depthwise convolution with residual connection
    '''
    # 一个加了残差连接的深度卷积（从groups=dim可以看出是每个通道独立计算的）
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
    
    def forward(self, x):
        x = x + super().forward(x)
        return x


class RepConvBlock(Module):
    # 对应于论文中的Basic Block
    def __init__(self, 
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False):
        super().__init__()
        
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        
        mlp_dim = int(dim*mlp_ratio)
        
        self.dwconv = ResDWConv(dim, kernel_size=3)
    
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            #### 这一部分是ConvFFN的结构
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            ####
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_init_value = ls_init_value
        # layerscale使用1x1 conv实现，输入维度为dim，输出维度为1
        
    def forward_features(self, x):
        
        x = self.dwconv(x)
        # 应该是残差连接与layerscale的两种不同的组合方式
        if self.res_scale:

            y = self.ls(x)
            z = self.proj(x)
            #print(x.shape, y.shape, z.shape)
            x = y + z
           
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))
        return x
    
    def execute(self, x):
        # 加了断点，应该是为了调试方便
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        
        return x




class DynamicConvBlock(nn.Module):
    def __init__(self,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 is_first=False,
                 is_last=False,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 **kwargs):
        
        super().__init__()
        
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim*mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)
        
        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)
        
        self.fusion = nn.Sequential( ##### DynamicBlock 中的ConvFFN ？？
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )
        
        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
         
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7), ######## 它处理小特征图可能有问题
            nn.Conv2d(ctx_dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
        
        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1) # 49 = 7 x 7 = S x S
        
        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        ) ######
        
        self.se_layer = SEModule(dim)
        
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )
        
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )
        
        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.get_rpb()


    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = jittor.empty(self.num_heads, self.rpb_size1, self.rpb_size1)
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = jittor.empty(self.num_heads, self.rpb_size2, self.rpb_size2)
        jittor.init.zero_(self.rpb1)
        jittor.init.zero_(self.rpb2)
    
        
    #@torch.no_grad()
    @jittor.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = jittor.arange(0, kernel_size)
        idx_w = jittor.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)
    

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """
        RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3
        """
        # attn shape: (B, g, H, W, K**2)
        # rpb shape: (G, K, K)
        
        # num_repeat_h = jittor.ones(kernel_size, dtype=jittor.int64)
        # num_repeat_w = jittor.ones(kernel_size, dtype=jittor.int64)
        # num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        # num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        
        # num_repeat_h的含义：
        # natten其实仍然是从KQV视角理解注意力计算，只不过对于每个查询像素q，将它的K和V的选择限制在q的邻域内，即一个卷积核的大小
        # 当q处在图像靠中间的位置时，以q的位置为中心的窗口（大小为卷积核尺寸）被完整包含在特征图中，这就是它的邻域；
        # 当q处在水平或竖直方向靠边的位置上时，以q为中心的窗口不能被完整包含在特征图中，所以
        # 就需要把卷积窗口往特征图内部推一点，造成q不再是窗口的中心，而是偏一点的位置
        # 这里的num_repeat实际就是统计水平与竖直方向上，q出现在其局部KV邻域窗口中不同相对位置分别的次数
        # 对于q出现在窗口中心的情况，即以q为中心的窗口完整包含于特征图的情况，
        # 恰在水平和竖直方向上分别重复出现W - K + 1 和 H - K + 1次（与卷积的滑动窗口个数一样）
        # 而q出现在窗口非中心位置的情况，恰好各有一次
        # jittor的repeat_interleave的repeats似乎只支持int，不支持对每个元素说明特定重复次数
        
        # 换一种实现，兼容jittor：
        repeated_idx_h = jittor.full(height, kernel_size // 2, dtype=jittor.int64)
        repeated_idx_h[0:kernel_size // 2] = jittor.arange(0, kernel_size // 2)
        repeated_idx_h[height - kernel_size // 2:height] = jittor.arange(kernel_size // 2 + 1, kernel_size)
        
        repeated_idx_w = jittor.full(width, kernel_size // 2, dtype=jittor.int64)
        repeated_idx_w[0:kernel_size // 2] = jittor.arange(0, kernel_size // 2)
        repeated_idx_w[width - kernel_size // 2:height] = jittor.arange(kernel_size // 2 + 1, kernel_size)
        bias_hw = (repeated_idx_h.unsqueeze(-1) * (2*kernel_size-1)) + repeated_idx_w
        # 上面这一步的解释：
        # 用repeat_interleave 分别计算
        bias_idx = bias_hw.unsqueeze(-1) + idx_k 
        # shape: (H, W, K ** 2)，代表H x W的特征图中每个位置作为query的时候，所需的K*K个bias在rpb参数中的索引
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))  # (H * W, K ** 2)
        bias_idx = jittor.flip(bias_idx, [0]) # 翻转存储的空间顺序
        rpb = jittor.flatten(rpb, 1, 2)[:, bias_idx] # 将可训练的rpb的后两个维度展平，得到shape为(G, K*K) （G是分组数或者说头数）
        # 这里使用bias_idx作为索引，应该就是用前面设置好的索引方案来用rpb填充整个图
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb
    def na2d_av(self, attn, value, kernel_size):
        # 尝试自己实现na2d_av
        # 思路：
        # 先把value用某种方式reindex，得到与attn匹配的形状
        # 然后再作简单的张量乘法（可能用einsum），得到结果
        
        # attn shape: (B, g, H, W, K**2)
        # value shape: (B, g, H, W, C)
        # target value shape: (B, g, H, W, K**2, C)
        B, G, H, W, _ = attn.shape
        K = kernel_size
        half_K = K // 2
        C = value.shape[-1]
        #neighbor_values = jittor.zeros([B, G, H, W, K, K, C]) # 尝试跳过会怎么样
        neighbor_values = jittor.reindex(
            value,
            shape = [B, G, H, W, K, K, C],
            indexes = [  ######## 基于target 索引表示出source的索引
                'i0', # B
                'i1', # G
                # 先考虑最简单的情形，即query在邻域中心的情况
                # f'i2 + i4 - {K // 2}',
                # f'i3 + i5 - {K // 2}',
                # 对于边缘情况，以水平方向为例：
                # 如果i3 - K // 2 < 0，则向左越界，所有的i都应该比理想情况向右偏移K//2 - i3
                # 所以就是要加上max(K // 2 - i3, 0)
                # 如果i3 + K // 2 > W - 1，则向右越界，所有的i都应该比理想情况向左偏移W - 1 - (i3 + K // 2)
                # 其实本质上是要对i3（窗口的中心）进行clamp操作，使它落在K // 2 <= i3 <= W - K // 2 - 1的范围内
                # 所以结果应该是：
                f'(((i2 >{half_K})?(i2):{half_K})<{H - half_K - 1})?((i2 >{half_K})?(i2):{half_K}):{H - half_K - 1} + i4 - {half_K}',
                f'(((i3 >{half_K})?(i3):{half_K})<{W - half_K - 1})?((i3 >{half_K})?(i3):{half_K}):{W - half_K - 1} + i5 - {half_K}',
                #f'min(max(i3, {half_K}), {W - half_K - 1}) + i5 - {half_K}',
                #f'clamp(i2, {half_K}, {H - half_K - 1}) + i4 - {half_K}',
                #f'clamp(i3, {half_K}, {W - half_K - 1}) + i5 - {half_K}',
                'i6'
            ]
        )
        neighbor_values = rearrange(neighbor_values, 'b g h w i j c -> b g h w (i j) c')
        #attn = rearrange(attn, 'b g h w l -> b g (h w) l')
        av = einsum('b g h w l, b g h w l c -> b g h w c', attn, neighbor_values)
        return av
    def _forward_inner(self, x, h_x, h_r):
        input_resoltion = x.shape[2:]   
        B, C, H, W = x.shape
        B, C_h, H_h, W_h = h_x.shape
        
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = jittor.cat([x, h_x], dim=1)
        #res dwconv
        x_f = self.dwconv1(x_f)
        # 后面这个残差块
        identity = x_f
        # GDSA
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x) ###### GDSA的gate分支
        lepe = self.lepe(x) ###### 包含一个扩张重参数块和一个batchnorm

        is_pad = False
        if min(H, W) < self.kernel_size:
            # 如果图像过小，就通过插值放大，确保最小的边长都不小于核的边长
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)
            x = nn.interpolate(x, size=size, mode='bilinear', align_corners=False)
            x_f = nn.interpolate(x_f, size=size, mode='bilinear', align_corners=False)
            H, W = size
        # contmix
        query, key = jittor.split(x_f, split_size=[C, C_h], dim=1)
        # 分别加工query和key
        query = self.weight_query(query) * self.scale
        #print("!!!", key.shape, self.weight_key[0])
        key = self.weight_key(key) ## 这里面包含了adaptive pooling
        # 分组
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        # 点积计算注意力
        weight = einsum('b g c n, b g c l -> b g n l', query, key) # g 即分组数，n = H*W，l = S*S
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        # 用1x1卷积，将weight的通道数l(=7x7=49)投影成大核与小核权重点数量之和。注意这里仍然保留了H和W
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)
        # 最后一个维度拆分为两种不同大小的核。注意这里仍然保留了H和W（这两个维度来自前面Q的计算）
        attn1, attn2 = jittor.split(weight, split_size=[self.smk_size**2, self.kernel_size**2], dim=-1)
        # rpb1_idx, rpb2_idx应该是某些索引信息
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        # 这里应该是在结合索引信息生成核权重
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        # attn1, attn2 shape: (B, g, H, W, K**2)
        # 这里的attn随着空间位置平移而发生变化，是因为引入了RPB（从而让不同位置具有基于相对位置计算得到的特定偏置）
        # 所以它不是简单的传统卷积，需要注意
        attn1 = nn.softmax(attn1, dim=-1)
        attn2 = nn.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
        # value[i] shape: (B, g, H, W, C)
        # 使用生成的核权重进行卷积
        ########## 原文使用了natten的实现，这里可能得自己重新写（尤其是需要注意：query在边缘时注意力窗口是向内回缩的）
        # value的窗口：在边缘处的K//2个位置的窗口都是一样的
        x1 = self.na2d_av(attn1, value[0], kernel_size=self.smk_size)
        x2 = self.na2d_av(attn2, value[1], kernel_size=self.kernel_size)
        # 合并大小核卷积结果
        x = jittor.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)

        if is_pad:
            self.adaptivePool = nn.AdaptiveAvgPool2d((input_resoltion[0], input_resoltion[1]))
            #print(self.kernel_size, input_resoltion)
            x = self.adaptivePool(x)

        x = self.dyconv_proj(x)

        x = x + lepe
        x = self.se_layer(x)

        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))
        
        x = self.dwconv2(x)
         
        if self.res_scale:###################### 定位到这里出错了
            y = self.norm2(x)
            y = self.mlp(y)
            y = self.drop_path(y)
            x = self.ls2(x)
            x = x + y
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.is_last:
            return (x, None)
        else:
            l_x, h_x = jittor.split(x, split_size=[C, C_h], dim=1)
            return (l_x, h_x)
    
    def execute(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        else:
            x = self._forward_inner(x, h_x, h_r)
        return x
class OverLoCK(Module):
    def __init__(
         self, 
         depth=[2, 2, 2, 2],
         sub_depth=[4, 2],
         in_chans=3, 
         embed_dim=[96, 192, 384, 768],
         kernel_size=[7, 7, 7, 7],
         mlp_ratio=[4, 4, 4, 4],
         sub_mlp_ratio=[4, 4],
         sub_num_heads=[4, 8],
         ls_init_value=[None, None, 1, 1],
         res_scale=True,
         smk_size=5,
         deploy=False,
         use_gemm=True,
         use_ds=True,
         drop_rate=0,
         drop_path_rate=0,
         norm_layer=LayerNorm2d,
         projection=1024,
         num_classes=1000,
         use_checkpoint=[0, 0, 0, 0],
            ):
        super().__init__()
        
        fusion_dim = embed_dim[-1] + embed_dim[-1]//4 # 融合的特征通道数
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv2d(embed_dim[-1], embed_dim[-1]//4, kernel_size=1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3]) # 这个是为focus-net准备的特制的下采样层（上下文也要处理）
        
        dpr = [x.item() for x in jittor.misc.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()
        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()
        # depth 应该是四个阶段每个阶段使用的blocks的数量
        for i in range(depth[0]):
            self.blocks1.append(
                RepConvBlock(
                    dim=embed_dim[0],
                    kernel_size=kernel_size[0],
                    mlp_ratio=mlp_ratio[0],
                    ls_init_value=ls_init_value[0],
                    res_scale=res_scale,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[0]),
                )
            )
        
        for i in range(depth[1]):
            self.blocks2.append(
                RepConvBlock(
                    dim=embed_dim[1],
                    kernel_size=kernel_size[1],
                    mlp_ratio=mlp_ratio[1],
                    ls_init_value=ls_init_value[1],
                    res_scale=res_scale,
                    drop_path=dpr[i+depth[0]],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[1]),
                )
            )
            
        for i in range(depth[2]):
            self.blocks3.append(
                RepConvBlock(
                    dim=embed_dim[2],
                    kernel_size=kernel_size[2],
                    mlp_ratio=mlp_ratio[2],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth[:2])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[2]),
                )
            )

        for i in range(depth[3]): # 第四个阶段应该是overview-net中的blocks
            self.blocks4.append(
                RepConvBlock(
                    dim=embed_dim[3],
                    kernel_size=kernel_size[3],
                    mlp_ratio=mlp_ratio[3],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth[:3])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[3]),
                )
            )
        # focus-net的动态块的堆叠。sub_depth就是对应的堆叠blocks数量
        for i in range(sub_depth[0]):
            self.sub_blocks3.append(
                DynamicConvBlock(
                    dim=embed_dim[2],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[2],
                    num_heads=sub_num_heads[0],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[0],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    is_first=(i==0),
                    use_checkpoint=(i<use_checkpoint[2]),
                )
            )
        
        for i in range(sub_depth[1]):
            self.sub_blocks4.append(
                DynamicConvBlock(
                    dim=embed_dim[3],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[-1],
                    num_heads=sub_num_heads[1],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[1],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)+sub_depth[0]],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    is_first=False,
                    is_last=(i==sub_depth[1]-1),
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[3]),
                )
            )

        # Aux Cls Head
        # 这个辅助分类头应该是图像分类预训练时接在overviewnet后的
        if use_ds:
            self.aux_head = nn.Sequential(
                nn.BatchNorm2d(embed_dim[-1]),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim[-1], num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
            )
        
        # Main Cls Head
        # 主要的分类头，应该是接在focus-net后的
        self.head = nn.Sequential(
            nn.Conv2d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
        )
        
        self.apply(self._init_weights)
        ##### 暂不考虑分布式训练
        #if torch.distributed.is_initialized():
        #    self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def reparm(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()
            
    def forward_pre_features(self, x):
        # 1,2阶段的计算
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
            
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        return x
    
    
    def forward_base_features(self, x):
        #3, 4阶段的计算
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
            
        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)

        return (x, ctx)
    

    def forward_sub_features(self, x, ctx):

        ctx_cls = ctx # 是overviewnet输出的1/32的小特征图
        ctx_ori = self.high_level_proj(ctx) 
        #ctx_ori的计算：压缩通道数量到原来的1/4，
        # 这样就可以直接拼接到第二个dynamic blocks阶段（图中弯曲长红线）,即self.sub_blocks4
        ctx_up = nn.interpolate(ctx_ori, size=x.shape[2:], mode='bilinear', align_corners=False)
        # ctx_up对ctx_ori进行了上采样，恢复到1/16尺寸的特征图（与basenet输出尺寸对齐），
        # 这样可以通过图中短直红线注入第一个dynamic blocks阶段 （self.sub_blocks3）
        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)

        x, ctx = self.patch_embedx(x, ctx)
        for idx, blk in enumerate(self.sub_blocks4):
            x, ctx = blk(x, ctx, ctx_ori)
        
        return (x, ctx_cls) # 这里返回的ctx_cls是overview得到的原始的上下文信息，未经过focus-net加工

    def forward_features(self, x):
        #  主要的特征计算过程都在这了
        x = self.forward_pre_features(x) #1, 2阶段的计算
        x, ctx = self.forward_base_features(x) # 3, 4阶段的计算（包括basenet的最后部分和overviewnet的计算）
        # 从图中可以看出，第三阶段的输出就是basenet输出的base_feature，即x
        # 第四阶段的输出是在base_feature上加工得到的context guidance，即ctx
        x, ctx_cls = self.forward_sub_features(x, ctx) # focus-net的计算

        return (x, ctx_cls)

    def execute(self, x):
        
        x, ctx = self.forward_features(x)
        x = self.head(x).flatten(1) #main 分类头计算

        if hasattr(self, 'aux_head') and self.training: #预训练时使用辅助分类头
            ctx = self.aux_head(ctx).flatten(1) # 这里用于aux_head计算的ctx是overview-net输出的原始上下文信息
            return dict(main=x, aux=ctx)
        
        return x
def _cfg(url=None, **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',  # 'bilinear' or 'bicubic'
        #'mean': timm.data.IMAGENET_DEFAULT_MEAN,
        #'std': timm.data.IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }

def overlock_xt(pretrained=False, pretrained_cfg=None, **kwargs):
    
    model = OverLoCK(
        depth=[2, 2, 3, 2],
        sub_depth=[6, 2],
        embed_dim=[56, 112, 256, 336],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 6],
        sub_mlp_ratio=[3, 3],
        **kwargs
    )

    model.default_cfg = _cfg(crop_pct=0.925)

    if pretrained:
        pretrained = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth'
        load_checkpoint(model, pretrained)

    return model
def overlock_xxt(pretrained=False, pretrained_cfg=None, **kwargs):
    """
    极小规模OverLoCK模型，参数和计算量约为overlock_xt的1/4-1/5
    适合移动端或边缘设备
    """
    model = OverLoCK(
        # depth=[2, 2, 3, 2],           # 最小深度配置
        # sub_depth=[6, 2],              # 最简动态块配置
        # embed_dim=[12, 24, 48, 64],   # 极低通道数
        # # embed_dim=[24, 48, 96, 128],   # 极低通道数 
        # kernel_size=[13, 11, 9, 7],      # 更小的卷积核 
        # #（kernel不能小于7x7，不然处理小特征图时，contimix中key的adaptive pooling会出错）
        # mlp_ratio=[2, 2, 2, 2],        # 最小MLP扩展
        # sub_num_heads=[1, 2],          # 最少注意力头
        # sub_mlp_ratio=[2, 2],
        # # projection=256,               # 极简分类头
        # projection=128,               # 极简分类头
        depth=[1, 1, 1, 1],           # 最小深度配置
        sub_depth=[2, 1],              # 最简动态块配置
        embed_dim=[24, 48, 96, 128],   # 极低通道数
        # embed_dim=[24, 48, 96, 128],   # 极低通道数
        kernel_size=[13, 11, 9, 7],      # 更小的卷积核
        mlp_ratio=[2, 2, 2, 2],        # 最小MLP扩展
        sub_num_heads=[1, 2],          # 最少注意力头
        sub_mlp_ratio=[2, 2],
        projection=256,               # 极简分类头
        #use_checkpoint=[0, 0, 0, 0],
        **kwargs
    )
    
    model.default_cfg = _cfg(crop_pct=0.85)

    if pretrained:
        print("Note: overlock_xxs currently has no pretrained weights")
        
    return model