# -*- coding: utf-8 -*-

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mlp(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.GELU,
                 drop = 0.):
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

class Attention(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads = 8,
                 qkv_bias = False,
                 qk_scale = None,
                 attn_drop = 0.,
                 proj_drop = 0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 mlp_ratio = 4.,
                 qkv_bias = False,
                 qk_scale = None,
                 drop = 0.,
                 attn_drop = 0., 
                 drop_path = 0.,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.  else nn.Identity()

        self.attn = Attention(dim, num_heads = num_heads, qkv_bias = qkv_bias,
                              qk_scale = qk_scale, attn_drop = attn_drop, proj_drop = drop)
        self.feedforward = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, 
                               act_layer = act_layer, drop = drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.feedforward(self.norm2(x)))

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, dim, in_chans = 3):
        super().__init__()
        self.patch_size = patch_size

        if not img_size % patch_size == 0:
            raise Exception("Image dimensions must be divisible by the patch size.")
        patch_dim = in_chans * patch_size ** 2

        self.fc = nn.Linear(patch_dim, dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        H_ = H // self.patch_size
        W_ = W // self.patch_size

        x = x.reshape(B, C, H_, self.patch_size, W_, self.patch_size).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, H_ *  W_, -1)
        x = self.fc(x)
        return x


class CLSToken(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    def forward(self, x):
        B, N, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim = 1)

        return x

class AbsPosEmbedding(nn.Module):
    def __init__(self, 
                 img_size,
                 patch_size,
                 dim,
                 stride = None,
                 cls = True):
        super().__init__()
        if not img_size % patch_size == 0:
            raise Exception("Image dimensions must be divisible by the patch size.")
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(img_size, patch_size, stride)
        num_patches = output_size ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + int(cls), dim) * 0.02)
    
    def forward(self, x):
        x = x + self.pos_embed
        return x
    
    @staticmethod
    def _conv_output_size(img_size, kernel_size, stride, padding = 0):
        return int(((img_size - kernel_size + (2 * padding)) / stride) + 1)

class ViT(nn.Module):
    def __init__(self,
                 img_size = 224,
                 in_chans = 3,
                 patch_size = 16,
                 num_classes = 100,
                 depth = 8,
                 dim = 128,
                 num_heads = 4,
                 mlp_ratio = 4,
                 qkv_bias = False,
                 qk_scale = None,
                 drop_rate = 0.2,
                 embed_drop_rate = 0.,
                 attn_drop_rate = 0.,
                 drop_path_rate = 0.1,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm):
        super().__init__()

        self.embedding = nn.Sequential(
            PatchEmbedding(img_size, patch_size, dim, in_chans = in_chans),
            CLSToken(dim),
            AbsPosEmbedding(img_size, patch_size, dim, cls = True),
            nn.Dropout(embed_drop_rate) if embed_drop_rate > 0. else nn.Identity()
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]   # stochastic depth decay rule

        self.blocks = nn.ModuleList([
                                  Transformer(dim = dim, num_heads = num_heads, mlp_ratio = mlp_ratio,
                                              qkv_bias = qkv_bias, qk_scale = qk_scale, drop = drop_rate,
                                              attn_drop = attn_drop_rate, drop_path = dpr[i], act_layer = act_layer,
                                              norm_layer = norm_layer) for i in range(depth)
        ])

        self.head = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        cls_token = x[:, 0]
        x = self.head(cls_token)
        return x


model = ViT(num_classes = 100).to(device)
