import sys
sys.path.append(r'D:\dl_project\dl_project_cnn\Backbone')
sys.path.append(r'D:\dl_project\dl_project_cnn\Utils')

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
import math


import math

#from Backbone.FirstStagenet import FirstStagenet_FE
from Utils.modules import Upsample
from Utils.modules import BasicPreActBlock
from Utils.modules import normer
from Utils.modules import Bottleneck
from Utils.modules import BasicBlock

class utils_classifier(nn.Module):
    def __init__(self, num_classes):
        super(utils_classifier, self).__init__()

        self.out = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)

    def forward(self, features):
        x = self.out(features['logit'])
        target_shape = x.shape[-2:] * 2
        x = F.interpolate(x, size = target_shape, mode='bilinear', align_corners=False)  # bilinear

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, _group_size=1, if_kv=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.if_kv = if_kv

        if not self.if_kv:
            logging.info(f'not kv')
            self.sr_ratio = sr_ratio
            if sr_ratio > 1:
                self.sr = nn.ModuleList([
                    nn.Sequential(
                        nn.BatchNorm2d(dim),
                        nn.ReLU(inplace=True),
                        # nn.Conv2d(dim, dim, kernel_size=2, stride=2),
                        nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
                        # logging.info(f'{0.5 * np.log2(sr_ratio)}')
                    ) for _ in range(int(np.log2(sr_ratio)))
                ])
            self.norm = nn.LayerNorm(dim)

        self._group_size = _group_size
        if self._group_size > 1:
            self.unfold = nn.Unfold(kernel_size=_group_size, stride=_group_size)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, kv=None):
        B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.if_kv:
            kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                           4)  # 2 B head L C
        else:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                for sr_blk in self.sr:
                    x_ = sr_blk(x_)
                x_ = x_.reshape(B, C, -1).permute(0, 2, 1)  # BLC
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2 B head L C

        if self._group_size > 1:
            x__ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x__ = rearrange(x__, 'b c (p1 h1) (p2 w1) -> b (p1 p2) (h1 w1) c', p1=self._group_size, p2=self._group_size)
            # x__ = self.unfold(x__).reshape(B, C, self._group_size**2, -1).permute(0, 2, 3, 1)  # B G L C
            x__ = self.norm(x__)
            q = self.q(x__) #  G B  HEAD L C
            q = rearrange(q, 'b g l (h1 c1) -> g b h1 l c1', h1= self.num_heads)  # GBHLC
        else:
            q = self.q(x)
            q = rearrange(q, 'b l (h1 c1) -> b h1 l c1', h1=self.num_heads)  # BHLC

        k, v = kv[0], kv[1]

        attn =(q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self._group_size > 1:
            x = (attn @ v) # G B HEAD lc
            x = rearrange(x, 'g b h l c -> b (g l) (h c)')
        else:
            x = (attn @ v).transpose(1, 2)  # B HEAD L C
            x = rearrange(x, 'b l h c -> b l (h c)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.dim = [in_chans, 64, 256]
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # self.proj = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv2d(self.dim[i], self.dim[i+1], kernel_size=4, stride=4),
        #             nn.BatchNorm2d(self.dim[i+1]),
        #             nn.ReLU(inplace=True)
        #         ) for i in range(int(np.log2(patch_size[0])//2))
        #     ])
        # self.proj = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # self.preproj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # x = self.preproj(x)
        # for pj in self.proj:
        #     x = pj(x)
        # x = self.proj(x)
        # H = self.img_size[0]//self.patch_size[0]
        # W = self.img_size[1]//self.patch_size[1]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # x = x.transpose(1, 2)
        # x = self.norm(x)

        return x, H, W

class EfficientSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=8, _group_size=8, if_kv = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, _group_size=_group_size, if_kv=if_kv)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, kv=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, kv=kv))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class Road_Reasoner(nn.Module):
    def __init__(self, dim=256, sr_ratio=2, numheads=1, if_kv=False):
        super(Road_Reasoner, self).__init__()
        self.OverlapPatchEmbed_1 = OverlapPatchEmbed(img_size=512, patch_size=16, stride=4, in_chans=1, embed_dim=dim)
        self.EfficientSelfAttention_1 = EfficientSelfAttention(dim=dim, num_heads=numheads, mlp_ratio=4, drop=0.,
                                                               attn_drop=0., sr_ratio=sr_ratio, _group_size=1, if_kv=if_kv)
        self.norm1 = nn.LayerNorm(dim)  # 32*32
        # self.EfficientSelfAttention_2 = EfficientSelfAttention(dim=dim, num_heads=1, mlp_ratio=4, drop=0.,
        #                                                        attn_drop=0., sr_ratio=sr_ratio[1], _group_size=1)
        # self.norm2 = nn.LayerNorm(dim)  # 32*32
        # self.EfficientSelfAttention_3 = EfficientSelfAttention(dim=dim, num_heads=1, mlp_ratio=4, drop=0.,
        #                                                        attn_drop=0., sr_ratio=sr_ratio[2], _group_size=1)
        # self.norm3 = nn.LayerNorm(dim)  # 32*32
        # # self.EfficientSelfAttention_4 = EfficientSelfAttention(dim=dim, num_heads=1, mlp_ratio=4, drop=0.,
        # #                                                        attn_drop=0., sr_ratio=sr_ratio[3], _group_size=1)
        # # self.norm4 = nn.LayerNorm(dim)  # 32*32
        # self.rf_attention = rf_attention(rf_num=3)
        # self.ca = ChannelAttention(in_channels=512)


    def forward(self, x, kv=None):
        # logging.info(f'start reasoning...')
        B = x.shape[0]
        # identity = x
        x, H, W = self.OverlapPatchEmbed_1(x)  # 16*16  64*64
        x = self.EfficientSelfAttention_1(x, H, W, kv=kv)
        x = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W 64 * 64
        #
        # x2 = self.EfficientSelfAttention_2(x, H, W)
        # x2 = self.norm2(x2).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W 64 * 64
        #
        # x3 = self.EfficientSelfAttention_3(x, H, W)
        # x3 = self.norm3(x3).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W 64 * 64

        # x4 = self.EfficientSelfAttention_4(x, H, W)  # B L C
        # x4 = self.norm4(x4).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W 64 * 64

        # out = self.rf_attention([x1,x2, x3])
        # x1_w, x2_w, x3_w= torch.split(out, split_size_or_sections=1, dim=1)
        # out = x1 * x1_w + x2 * x2_w +  x3 * x3_w

        # out = self.ca(out) * out

        # return out + identity
        return x


class GRRM(nn.Module):
    def __init__(self, dim_ec=256, dim_dm=256, tg_dim=256, size=32*32):
        super(GRRM, self).__init__()
        # self.OverlapPatchEmbed_1 = OverlapPatchEmbed(img_size=512, patch_size=16, stride=4, in_chans=1,
        #                                              embed_dim=dim)
        # self.EfficientSelfAttention_1 = EfficientSelfAttention(dim=dim, num_heads=numheads, mlp_ratio=4, drop=0.,
        #                                                        attn_drop=0., sr_ratio=sr_ratio, _group_size=1,
        #                                                        if_kv=if_kv)
        self.topo_enc = nn.Conv2d(dim_ec, tg_dim, 1)
        self.topo_dpmp=nn.Conv2d(dim_dm, tg_dim, 1)
        self.norm_dec = nn.LayerNorm(tg_dim)  # 32*32
        self.norm_dpmp = nn.LayerNorm(tg_dim)  # 32*32
        self.norm_pm = nn.LayerNorm(size)
        # self.scale = tg_dim ** -0.5
        self.conv0 = nn.Sequential(
            nn.Conv2d(size, tg_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(tg_dim, tg_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(tg_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(tg_dim, tg_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(tg_dim),
        )
        # self.conv1= BasicBlock(tg_dim, tg_dim, stride=1, norm_layer=nn.BatchNorm2d)
        self.relu = nn.ReLU(inplace=True)
        self.skip_cn = nn.Conv2d(dim_ec, tg_dim, 1)

        # self.EfficientSelfAttention_2 = EfficientSelfAttention(dim=dim, num_heads=1, mlp_ratio=4, drop=0.,
        #                                                        attn_drop=0., sr_ratio=sr_ratio[1], _group_size=1)
        # self.norm2 = nn.LayerNorm(dim)  # 32*32
        # self.EfficientSelfAttention_3 = EfficientSelfAttention(dim=dim, num_heads=1, mlp_ratio=4, drop=0.,
        #                                                        attn_drop=0., sr_ratio=sr_ratio[2], _group_size=1)
        # self.norm3 = nn.LayerNorm(dim)  # 32*32
        # # self.EfficientSelfAttention_4 = EfficientSelfAttention(dim=dim, num_heads=1, mlp_ratio=4, drop=0.,
        # #                                                        attn_drop=0., sr_ratio=sr_ratio[3], _group_size=1)
        # # self.norm4 = nn.LayerNorm(dim)  # 32*32
        # self.rf_attention = rf_attention(rf_num=3)
        # self.ca = ChannelAttention(in_channels=512)



    def forward(self, ec, dm):
        # logging.info(f'start reasoning...')
        identity =  self.skip_cn(ec)
        B, _, H, W = ec.shape
        # sf = dm.flatten(2).transpose(1, 2)  # BLC
        e = self.topo_enc(ec) # proj
        e = e.flatten(2).transpose(1, 2) # reshape BLC
        
        m = self.topo_dpmp(dm)
        m = self.norm_dpmp(m.flatten(2).transpose(1, 2))# reshape  # BLC
        poe_map = self.norm_dec(e) @ m.transpose(-2, -1)
        # tp = tp*self.scale
        # tpr = tp.softmax(dim=-1)  # tp relationship
        # new_map = tpr @ sf
        poe_map = self.norm_pm(poe_map)
        poe_map = poe_map.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.relu(self.conv0(poe_map) + identity)
        return x


class training_classifier(nn.Module):
    def __init__(self, num_classes, norm_layer = None):
        super(training_classifier, self).__init__()

        self.upsample_1 = nn.Sequential(
            # Road_Reasoner(dim=512, sr_ratio=1, numheads=1),
            Upsample(in_channels=512, out_channels=256, kernel=2, stride=2),
            # nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            # BasicBlock(in_channels=512, out_channels=256, norm_layer=norm_layer),

            # BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),

        )  # 太大了会淹没底层信息 128*128*256

        self.conv_1 = nn.Sequential(
            BasicBlock(in_channels=512, out_channels=256, norm_layer=norm_layer)
            # BasicBlock(in_channels=256, out_channels=256, norm_layer=norm_layer)
            # nn.Conv2d(512, 256, kernel_size=3, padding=1),
            # Road_Reasoner(dim=256, sr_ratio=2, numheads=1)
        )

        # self.rr1 = Road_Reasoner(dim=256, sr_ratio=2, numheads=1, if_kv=True)
        # self.kv_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        # self.norm_kv1 = nn.LayerNorm(256)
        self.tr1 = GRRM(dim_ec=256, dim_dm=512, tg_dim=256, size=32*32)

        self.upsample_2 = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Upsample(in_channels=256, out_channels=128, kernel=2, stride=2),
            norm_layer(128),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            # BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),

            # BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),

        )  # 太大了会淹没底层信息256*256*128

        self.conv_2 = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),
            # BasicBlock(in_channels=128, out_channels=128, norm_layer=norm_layer),
            # nn.Conv2d(256, 128, kernel_size=3, padding=1),
            # Road_Reasoner(dim=128, sr_ratio=4, numheads=1)

            # BasicBlock(in_channels=128, out_channels=64, norm_layer=norm_layer),

            # BasicBlock(in_channels=128, out_channels=64, norm_layer=norm_layer),
        )

        # self.rr2 = Road_Reasoner(dim=128, sr_ratio=4, numheads=1, if_kv=True)
        # self.kv_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        # self.norm_kv2 = nn.LayerNorm(128)
        self.tr2 = GRRM(dim_ec=128, dim_dm=512, tg_dim=128, size=32*32)

        self.upsample_3 = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Upsample(in_channels=128, out_channels=64, kernel=2, stride=2),
            norm_layer(64),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            # BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),

            # BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),

        )  # 太大了会淹没底层信息512*512*64

        self.hr_feature_extraction = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=64, norm_layer=norm_layer),
            BasicBlock(in_channels=64, out_channels=64, norm_layer=norm_layer),
        )

        self.conv_3 = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=64, norm_layer=norm_layer),
            # BasicBlock(in_channels=64, out_channels=64, norm_layer=norm_layer),
            # nn.Conv2d(128, 64, kernel_size=3, padding=1),
            # norm_layer(64),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 1, kernel_size=1),
            # # Road_Reasoner(dim=64, sr_ratio=8, numheads=1)

            # BasicBlock(in_channels=64, out_channels=64, norm_layer=norm_layer),

            # BasicBlock(in_channels=128, out_channels=64, norm_layer=norm_layer),
        )  # 512*512*64

        # self.rr3 = Road_Reasoner(dim=64, sr_ratio=8, numheads=1, if_kv=True)
        # self.kv_3 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        # self.norm_kv3 = nn.LayerNorm(64)
        # self.tr3 = GRRM(dim_ec=64, dim_dm=512, tg_dim=64, size=32*32)

        self.upsample_4 = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            Upsample(in_channels=64, out_channels=64, kernel=4, stride=2, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True)
            # BasicBlock(in_channels=64, out_channels=64, norm_layer=norm_layer),
            # nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            # BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),

            # BasicBlock(in_channels=256, out_channels=128, norm_layer=norm_layer),

        )  # 太大了会淹没底层信息512*512*64

        # self.skip_connect_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        # )

        # self.skip_connect_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        # )

        # self.skip_connect_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # )

        # self.out = nn.Sequential(
        #     # nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1),
        #     # norm_layer(32),
        #     # nn.ReLU(inplace=True),
        #     BasicBlock(in_channels=128, out_channels=32, norm_layer=norm_layer),
        #
        #     nn.Conv2d(32, num_classes, 1),
        #     # nn.Sigmoid(),
        # )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):

        kv = features['logit'].contiguous()
        # print(f'features:{features["logit"].data}')

        up1 = self.upsample_1(features['logit'])

        tp1 = self.tr1(features['1/4_res'], kv)

        # cat1 = torch.cat([features['1/4_res'], up1+ tp1], dim=1)
        cat1 = torch.cat([tp1, up1], dim=1)

        x1 = self.conv_1(cat1)
        # x1 = self.conv_1(up1)
        # x1 = self.rr1(x1, kv=self.norm_kv1(self.kv_1(kv).flatten(2).transpose(1, 2)))

        up2 = self.upsample_2(x1)

        tp2 = self.tr2(features['1/2_res'], kv)

        cat2 = torch.cat([tp2, up2], dim=1)  #
        # cat2 = torch.cat([features['1/2_res'], up2+ tp2], dim=1)

        # x2 = self.conv_2(up2)
        x2 = self.conv_2(cat2)
        # x2 = self.rr2(x2, kv=self.norm_kv2(self.kv_2(kv).flatten(2).transpose(1, 2)))

        up3 = self.upsample_3(x2)

        # tp3 = self.tr3(features['raw_res'], kv)
        #
        hr_feature = self.hr_feature_extraction(features['raw_res'])
        
        cat3 = torch.cat([hr_feature, up3], dim=1)  #
        # cat3 = torch.cat([tp3, up3], dim=1)
        # cat3 = torch_cat([features['raw_res'], up3], dim=1)
        x4 = self.conv_3(cat3)
        # x3 = self.rr3(x3, kv=self.norm_kv3(self.kv_3(kv).flatten(2).transpose(1, 2)))
        x4 = self.upsample_4(x4)

        # out = self.conv_2(x3)

        # out = self.out(out)

        return x4


