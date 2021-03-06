# -*- coding: utf-8 -*-

import math
import time
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# device configuration
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')


class ConvEmbedding(nn.Module):
    def __init__(self, 
                 patch_size = 7, 
                 stride = 4, 
                 in_chans = 3, 
                 embed_dim = 64):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size = patch_size, 
                              stride = stride,
                              padding = (patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape

        x = x.reshape(B, C, -1).transpose(-2, -1)
        x = self.norm(x)
        return x, H, W


class LocallyEnhancedAttention(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 qkv_bias = False,
                 attn_drop = 0.,
                 proj_drop = 0.,
                 kernel_size = 3,
                 sr_ratio = 7,
                 h_ratio = 0.5,
                 upsample_mode = "bilinear",):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.attention_map = None

        # split
        h_dim = int(dim * h_ratio)
        l_dim = dim - h_dim
        self.h_dim = h_dim
        self.l_dim = l_dim
        
        # for high-frequency feature
        self.proj_v = nn.Linear(h_dim, l_dim, bias = qkv_bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(l_dim, l_dim, kernel_size = kernel_size, stride = 1, 
                      padding = kernel_size // 2, bias = qkv_bias, groups = l_dim),
            nn.BatchNorm2d(l_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l_dim, l_dim, kernel_size = kernel_size, stride = 1, 
                      padding = kernel_size // 2, bias = qkv_bias, groups = l_dim),
            nn.BatchNorm2d(l_dim),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(l_dim, h_dim, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(h_dim),
        )
        self.gelu = nn.GELU()


        # q, k projection using pooling
        self.avg_pool_k = nn.AvgPool2d(kernel_size = sr_ratio)
        self.max_pool_k = nn.MaxPool2d(kernel_size = sr_ratio)
        self.avg_pool_q = nn.AvgPool2d(kernel_size = sr_ratio)
        self.max_pool_q = nn.MaxPool2d(kernel_size = sr_ratio)
        


        # For upsampling attention map
        self.k_upsample = nn.Upsample(scale_factor = sr_ratio, mode = upsample_mode)
        
        self.v_upsample = nn.Upsample(scale_factor = sr_ratio, mode = upsample_mode)
        self.v_conv = nn.Conv2d(l_dim, l_dim, kernel_size = kernel_size, stride = 1,
                                padding = kernel_size // 2, groups = l_dim, bias = False)
        self.v_bn = nn.BatchNorm2d(l_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x):
        B, N, C = x.shape
        h, w = int(math.sqrt(N)), int(math.sqrt(N))

        # split
        x_high, x_low = torch.split(x, [self.h_dim, self.l_dim], dim = -1)
     
        v = self.proj_v(x_high)
        x_high = v
        x_high = x_high.reshape(B, h, w, -1).permute(0, 3, 1, 2)
        x_high = self.conv1(x_high)
        q1 = self.avg_pool_q(x_high)   # B x C x h' x w'
        q2 = self.max_pool_q(x_high)      
        x_high = self.gelu(x_high)
        x_high = self.conv2(x_high)
        k1 = self.avg_pool_k(x_high)
        k2 = self.max_pool_k(x_high)
        x_high = self.conv3(x_high)
        
        x_high = self.gelu(x_high)
        x_high = x_high.reshape(B, self.h_dim, -1).transpose(-2, -1)

        q1 = q1.reshape(B, self.num_heads, self.l_dim // self.num_heads, -1).transpose(-2, -1)
        q2 = q2.reshape(B, self.num_heads, self.l_dim // self.num_heads, -1).transpose(-2, -1)
        q = torch.cat([q1, q2], dim = -1)

        k1 = k1.reshape(B, self.num_heads, self.l_dim // self.num_heads, -1).transpose(-2, -1)
        k2 = k2.reshape(B, self.num_heads, self.l_dim // self.num_heads, -1).transpose(-2, -1)
        k = torch.cat([k1, k2], dim = -1)
        
        v = v.reshape(B, N, self.num_heads, self.l_dim // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1))   # B x num_heads x N_ x N_ (N_ = h' * w')

        # Upsample attention map
        N_ = attn.shape[-1]
        h_, w_ = int(math.sqrt(N_)), int(math.sqrt(N_))
        attn = attn.reshape(-1, N_, int(math.sqrt(N_)), int(math.sqrt(N_)))
        attn = self.k_upsample(attn) * self.scale
        attn = attn.reshape(B, self.num_heads, N_, N)

        attn = attn.softmax(dim = -1)
        self.attention_map = attn
        attn = self.attn_drop(attn)
        
        x_low = (attn @ v).reshape(B, self.num_heads, h_, w_, -1)
        x_low = x_low.permute(0, 1, 4, 2, 3).reshape(B, -1, h_, w_)
        x_low = self.v_bn(self.v_conv(self.v_upsample(x_low)))
        x_low = x_low.reshape(B, self.l_dim, -1).transpose(-2, -1)

        x = torch.cat([x_high, x_low], dim = -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.GELU,
                 drop = 0.,
                 kernel_size = 3,
                 with_bn = True):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # Pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size = 1,
                               stride = 1, padding = 0)
        
        # Depthwise
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size = kernel_size,
                               stride = 1, padding = kernel_size // 2, groups = hidden_features)
        
        # Pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size = 1,
                               stride = 1, padding = 0)
        
        self.act =  act_layer()
        self.drop = nn.Dropout(drop)
        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), C).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.conv3(x)
            
        x = self.drop(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 mlp_ratio = 4,
                 qkv_bias = False,
                 drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 upsample_mode = "bilinear",
                 sr_ratio = 7.,
                 h_ratio = 0.5,
                 kernel_size = 3,
                 with_bn = True):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.leatt = LocallyEnhancedAttention(dim, num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop,
                                              proj_drop = drop, kernel_size = kernel_size, sr_ratio = sr_ratio,
                                              h_ratio = h_ratio, upsample_mode = upsample_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.leff = LocallyEnhancedFeedForward(in_features = dim, hidden_features = dim * mlp_ratio,
                                               act_layer = act_layer, drop = drop, 
                                               kernel_size = kernel_size, with_bn = with_bn)
    
    def forward(self, x):
        x = x + self.drop_path(self.leatt(self.norm1(x)))
        x = x + self.drop_path(self.leff(self.norm2(x)))
        return x


class LocallyEnhancedConvTransformer(nn.Module):
    def __init__(self, 
                 img_size = 224, 
                 patch_size = 7,
                 in_chans = 3,
                 kernel_size = 3,
                 num_classes = 100,
                 embed_dims = [64, 128, 192],
                 num_heads = [2, 3, 7],
                 mlp_ratios = [2, 3, 4],
                 h_ratios = [1/2 , 1/4, 1/8],
                 qkv_bias = False,
                 drop_rate = 0.1,
                 attn_drop_rate = 0.,
                 drop_path_rate = 0.1,
                 depths = [1, 3, 1],
                 num_stages = 3,
                 act_layer = nn.GELU,
                 norm_layer = partial(nn.LayerNorm, eps = 1e-6),
                 upsample_mode = "bilinear",
                 with_bn = True):
        """
        args:
            - img_size (:obj:'int') : input image size
            - patch_size (:obj:'int') : patch size
            - in_chans (:obj:'int') : input channels
            - kernel_size (:obj:'int') : kernel size for conv in attention module
            - num_classes (:obj:''int) : number of classes
            - embed_dims (:obj:`list`): list of embeddings dimensions for tokens
            - num_heads (:obj:`list`): list of numbers of heads in multi-head self-attention
            - mlp_ratios (:obj:`list`): list of expand ratios in feedforward
            - h_ratios (:obj:'list) : list of channel ratios for high-frequency feature
            - qkv_bias (:obj:`bool`): whether to add bias for mlp of qkv
            - drop_rate (:obj:`float`): dropout rate in feedforward module after linear operation
                and projection drop rate in attention
            - attn_drop_rate (:obj:`float`): dropout rate for attention
            - drop_path_rate (:obj:`float`): drop_path rate after attention
            - depths (:obj: 'list') : list of depth for each stage
            - num_stages (:obj:'int') : number of stage
            - act_layer (:obj:'nn.Module') : activation function type
            - norm_layer (:obj:`nn.Module`): normalization type
            - upsample_mode (:obj:'string') : upsample mode used in LeAtt
            - with_bn (:obj:'bool') : whether add bn in LocallyEnhancedFeedForward
        """
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]   # Stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = ConvEmbedding(patch_size = patch_size if i == 0 else 3,
                                        stride = 4 if i == 0 else 2,
                                        in_chans = 3 if i == 0 else embed_dims[i - 1],
                                        embed_dim = embed_dims[i])
            blocks = nn.ModuleList([
                                    Block(embed_dims[i], num_heads = num_heads[i],
                                          mlp_ratio = mlp_ratios[i], qkv_bias = qkv_bias,
                                          drop = drop_rate, attn_drop = attn_drop_rate,
                                          drop_path = dpr[cur + j], act_layer = act_layer,
                                          norm_layer = norm_layer, upsample_mode = "bicubic" if i == 0 else "bilinear",
                                          sr_ratio = 14 if i == 0 else 7, h_ratio = h_ratios[i],
                                          kernel_size = kernel_size, with_bn = with_bn) 
                                    for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"blocks{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)
        
        # Classifier Head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool = ''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0. else nn.Identity()
    
    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"blocks{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for block in blocks:
                x = block(x)
            x = norm(x)

            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim = 1)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

model = LocallyEnhancedConvTransformer(num_classes = 10).to(device)
