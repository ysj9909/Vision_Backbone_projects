# -*- coding: utf-8 -*-

import torch
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)



def conv1x1(in_channels, out_channels, stride = 1, groups = 1):
    return convnxn(in_channels, out_channels, kernel_size = 1, stride = stride, groups = groups)

def conv3x3(in_channels, out_channels, stride = 1, groups = 1):
    return convnxn(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, groups = groups)

def convnxn(in_channels, out_channels, kernel_size, stride = 1, groups = 1, padding = 0):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride ,
                     padding = padding, groups = groups, bias = False)

class GAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.gap(x)
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        return x



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 in_channels,
                 channels,
                 stride = 1,
                 groups = 1,
                 width_per_group = 64,
                 sd = 0.,
                 **kwargs):
        super().__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(conv1x1(in_channels, channels * self.expansion, stride = stride))
            self.shortcut.append(nn.BatchNorm2d(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = nn.Sequential(
            conv3x3(in_channels, width, stride = stride),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            conv3x3(width, channels * self.expansion),
            nn.BatchNorm2d(channels * self.expansion),
        )

        self.relu = nn.ReLU()
        self.sd = DropPath(sd) if sd > 0. else nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sd(x) + skip
        x = self.relu(x)
        return x



class ResNet(nn.Module):
    def __init__(self, 
                 block,
                 num_blocks,
                 cblock = GAPBlock,
                 sd = 0., 
                 num_classes = 10,
                 stem = True,
                 name = "resnet",
                 **block_kwargs):
        super().__init__()

        self.name = name
        idxs = [[j for j in range(sum(num_blocks[:i]), sum(num_blocks[:i + 1]))] for i in range(len(num_blocks))]
        sds = [[sd * j /(sum(num_blocks) - 1) for j in js] for js in idxs]

        self.layer0 = []
        if stem:
            self.layer0.append(convnxn(3, 32, kernel_size = 7, stride = 2, padding = 3))
            self.layer0.append(nn.BatchNorm2d(32))
            self.layer0.append(nn.ReLU())
            self.layer0.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        else:
            self.layer0.append(conv3x3(3, 32, stride = 1))
            self.layer0.append(nn.BatchNorm2d(32))
            self.layer0.append(nn.ReLU())
        
        self.layer0 = nn.Sequential(*self.layer0)

        self.layer1 = self._make_layer(block, 32, 32,
                                       num_blocks[0], stride=1, sds=sds[0], **block_kwargs)
        self.layer2 = self._make_layer(block, 32 * block.expansion, 64,
                                       num_blocks[1], stride=2, sds=sds[1], **block_kwargs)
        self.layer3 = self._make_layer(block, 64 * block.expansion, 96,
                                       num_blocks[2], stride=2, sds=sds[2], **block_kwargs)
        self.layer4 = self._make_layer(block, 96 * block.expansion, 128,
                                       num_blocks[3], stride=2, sds=sds[3], **block_kwargs)
        
        self.classifier = []
        self.classifier.append(cblock(128 * block.expansion, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)
    
    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks, stride , sds, **block_kwargs):
        stride_seq = [stride] + [1] * (num_blocks - 1)
        layer_seq, channels = [], in_channels
        for i in range(num_blocks):
            layer_seq.append(block(channels, out_channels, stride = stride_seq[i], 
                                   sd = sds[i], **block_kwargs))
            channels = out_channels * block.expansion
        return nn.Sequential(*layer_seq)
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = 10).to(device)
