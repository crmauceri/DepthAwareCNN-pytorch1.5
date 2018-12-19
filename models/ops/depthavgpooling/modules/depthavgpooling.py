import math, warnings

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from ..functions import depth_avgpooling


class Depthavgpooling(Module):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(Depthavgpooling, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        if not torch.cuda.is_available():
            warnings.warn("Warning: Not using depth")
            self.pool = torch.nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, input, depth):
        if not torch.cuda.is_available():
            return self.pool(input)
        else:
            return depth_avgpooling(input, depth, self.kernel_size, self.stride, self.padding)
