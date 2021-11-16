"""Code based on
https://raw.githubusercontent.com/yzou2/CRST/fce34003dd29c5f12f39d1c228dacd6277a064ae/deeplab/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
affine_par = True

__all__ = ['Res_Deeplab']

BatchNorm = nn.BatchNorm2d #nn.SyncBatchNorm

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self)