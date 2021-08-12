"""Code based on
https://raw.githubusercontent.com/yzou2/CRST/fce34003dd29c5f12f39d1c228dacd6277a064ae/deeplab/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
affine_par = True

