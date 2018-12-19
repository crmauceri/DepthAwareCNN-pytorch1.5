import math, warnings

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from ..functions import depth_conv


class DepthConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super(DepthConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        if not torch.cuda.is_available():
            warnings.warn("Warning: Not using depth")
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation, bias=bias)
        else:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, *self.kernel_size))

            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, depth):
        if not torch.cuda.is_available():
            return self.conv(input)
        else:
            return depth_conv(input, depth, self.weight, self.bias, self.stride,
                             self.padding, self.dilation)
