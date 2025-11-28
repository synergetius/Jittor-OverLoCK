import jittor
from jittor import Module
from jittor import nn
import numpy as np
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
        
    def forward(self, x):
        x = x * self.proj(x)
        return x
class LayerScale(Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = jittor.ones(dim, 1, 1, 1)*init_value
        self.bias = jittor.zeros(dim)

    def forward(self, x):
        x = nn.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
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

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
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

    def forward(self, x):
        if not hasattr(self, 'origin_bn'): # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
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

    def forward(self, x, ctx):
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
        # layerscale使用1x1 conv实现，输入维度为dim，输出维度为1
        
    def forward_features(self, x):
        
        x = self.dwconv(x)
        # 应该是残差连接与layerscale的两种不同的组合方式
        if self.res_scale: 
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x
    
    def forward(self, x):
        # 加了断点，应该是为了调试方便
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        
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
        
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
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
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode='bilinear', align_corners=False)
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

    def forward(self, x):
        
        x, ctx = self.forward_features(x)
        x = self.head(x).flatten(1) #main 分类头计算

        if hasattr(self, 'aux_head') and self.training: #预训练时使用辅助分类头
            ctx = self.aux_head(ctx).flatten(1) # 这里用于aux_head计算的ctx是overview-net输出的原始上下文信息
            return dict(main=x, aux=ctx)
        
        return x
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