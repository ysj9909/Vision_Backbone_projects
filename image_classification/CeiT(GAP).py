# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Image2Tokens(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 128, kernel_size = 7, stride = 2):
        super().__init__()

        # feature resolution을 절반으로 줄인다.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
                              stride = stride, padding = kernel_size // 2, bias = False)
        # Training을 용이하게 해준다.
        self.bn = nn.BatchNorm2d(out_channels)

        # feature map의 해상도를 한 번더 반으로 줄인다.
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        out_features = in_features or out_features
        hidden_features = in_features or hidden_features

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


class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, 
                 drop = 0., kernel_size = 3, with_bn = True):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # Pointwise conv 1 x 1 -> amplify num features so enlarge model's capacity
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size = 1, stride = 1, padding = 0)
        # Depthwise conv for CNN like operation
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size = kernel_size,
                               stride = 1, padding = (kernel_size - 1) // 2, groups = hidden_features)
        # Pointwise conv 1 x  1 -> get back to input's number of features
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size = 1, stride = 1, padding = 0)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)


        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    
    def forward(self, x):
        b, n, c = x.size()

        cls_token, tokens = torch.split(x, [1, n - 1], dim = 1)
        x = tokens.reshape(b, int(math.sqrt(n - 1)), int(math.sqrt(n - 1)), c).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)
        
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat([cls_token, tokens], dim = 1)

        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -.5

        self.qkv = nn.Linear(dim, 3 * dim, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attention_map = None

    def forward(self, x):
        b, n, c = x.size()
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        self.attention_map = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AttentionLCA(Attention):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0., ):
        super(AttentionLCA, self).__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.qkv_bias = qkv_bias

    def forward(self, x):
        
        q_weight = self.qkv.weight[:self.dim, :]
        q_bias = None if not self.qkv_bias else self.qkv.bias[:self.dim]
        kv_weight = self.qkv.weight[self.dim:, :]
        kv_bias = None if not self.qkv_bias else self.qkv.bias[self.dim:]

        b, n, c = x.size()

        _, last_token = torch.split(x, [n - 1, 1], dim = 1)

        q = F.linear(last_token, q_weight, q_bias).reshape(b, 1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv = F.linear(x, kv_weight, kv_bias).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, 1, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4, qkv_bias = False, qk_scale = None, drop = 0., attn_drop = 0.,
                 drop_path = 0., act_layer = nn.GELU, norm_layer = nn.LayerNorm, kernel_size = 3, with_bn = True,
                 feedforward_type = 'leff'):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else  nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.feedforward_type = feedforward_type

        if feedforward_type == 'leff':
            self.attn = Attention(
                dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale, attn_drop = attn_drop, proj_drop = drop
            )
            self.leff = LocallyEnhancedFeedForward(
                in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop,
                kernel_size = kernel_size, with_bn = with_bn
            )
        else:   # LCA
            self.attn = AttentionLCA(
                dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale, attn_drop = attn_drop, proj_drop = drop
            )
            self.feedforward = Mlp(
                in_features = dim,  hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop
            )
    
    def forward(self, x):
        if self.feedforward_type == 'leff':
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.leff(self.norm2(x)))
            return x, x[:, 0]
        else:
            _, last_token = torch.split(x, [x.size(1) -1, 1], dim = 1)
            x = last_token + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.feedforward(self.norm2(x)))
            return x

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding"""
    def __init__(self, backbone, img_size = 224, patch_size = 4, feature_size = None, in_chans = 3, embed_dim = 128):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                    o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                    if isinstance(o, (list, tuple)):
                        o = o[-1]   # last feature if backbone outputs list/tuple of features
                    feature_size = o.shape[-2:]
                    feature_dim = o.shape[1]
                    backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        print('feature_size is {}, feature_dim is {}, patch_size is {}'.format(
            feature_size, feature_dim, patch_size
        ))
         
        self.num_patches = (feature_size[0] // patch_size) * (feature_size[1] // patch_size)
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size = patch_size, stride = patch_size)

    
    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x  = x[-1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CeIT(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 4,
                 in_chans = 3,
                 num_classes = 10,
                 embed_dim = 128,
                 depth = 12,
                 num_heads = 4,
                 mlp_ratio = 4,
                 qkv_bias = False,
                 qk_scale = None,
                 drop_rate = 0.,
                 attn_drop_rate = 0.,
                 drop_path_rate = 0.,
                 hybrid_backbone = None,
                 norm_layer = nn.LayerNorm,
                 leff_local_size = 3,
                 leff_with_bn = True):
        """
        args:
            - img_size (:obj:`int`): input image size
            - patch_size (:obj:`int`): patch size
            - in_chans (:obj:`int`): input channels
            - num_classes (:obj:`int`): number of classes
            - embed_dim (:obj:`int`): embedding dimensions for tokens
            - depth (:obj:`int`): depth of encoder
            - num_heads (:obj:`int`): number of heads in multi-head self-attention
            - mlp_ratio (:obj:`float`): expand ratio in feedforward
            - qkv_bias (:obj:`bool`): whether to add bias for mlp of qkv
            - qk_scale (:obj:`float`): scale ratio for qk, default is head_dim ** -0.5
            - drop_rate (:obj:`float`): dropout rate in feedforward module after linear operation
                and projection drop rate in attention
            - attn_drop_rate (:obj:`float`): dropout rate for attention
            - drop_path_rate (:obj:`float`): drop_path rate after attention
            - hybrid_backbone (:obj:`nn.Module`): backbone e.g. resnet
            - norm_layer (:obj:`nn.Module`): normalization type
            - leff_local_size (:obj:`int`): kernel size in LocallyEnhancedFeedForward
            - leff_with_bn (:obj:`bool`): whether add bn in LocallyEnhancedFeedForward
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.i2t = HybridEmbed(
            hybrid_backbone, img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim
        )
        num_patches = self.i2t.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p = drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]   # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                                     Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias,
                                           qk_scale = qk_scale, drop = drop_rate , attn_drop = attn_drop_rate, 
                                           drop_path = dpr[i], norm_layer = norm_layer,
                                           kernel_size = leff_local_size, with_bn = leff_with_bn) for i in range(depth)
        ])
        
        # without droppath
        self.lca = Block(
            dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, qk_scale = qk_scale, 
            drop = drop_rate, attn_drop = attn_drop_rate, drop_path = 0., norm_layer = norm_layer,
            feedforward_type = 'lca'
        )
        self.pos_layer_embed = nn.Parameter(torch.zeros(1, depth, embed_dim))

        self.norm = norm_layer(embed_dim)

        # Classifier Head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std = 0.02)
        trunc_normal_(self.cls_token, std = 0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, blobal_pool = ''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.i2t(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim = 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        cls_token_list = []
        for blk in self.blocks:
            x, curr_cls_token = blk(x)
            cls_token_list.append(curr_cls_token)
        
        all_cls_token = torch.stack(cls_token_list, dim = 1)   # B * D * K
        all_cls_token = all_cls_token + self.pos_layer_embed

        # Attention over cls tokens
        last_cls_token = self.lca(all_cls_token)
        last_cls_token = self.norm(last_cls_token)

        return last_cls_token.view(B, -1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x 

i2t = Image2Tokens()
model = CeIT(img_size = 224,
                 patch_size = 4,
                 in_chans = 3,
                 num_classes = 10,
                 embed_dim = 112,
                 depth = 8,
                 num_heads = 4,
                 mlp_ratio = 2,
                 qkv_bias = True,
                 qk_scale = None,
                 drop_rate = 0.,
                 attn_drop_rate = 0.,
                 drop_path_rate = 0.1,
                 hybrid_backbone = i2t,
                 norm_layer = partial(nn.LayerNorm, eps = 1e-6),
                 leff_local_size = 3,
                 leff_with_bn = True).to(device)
