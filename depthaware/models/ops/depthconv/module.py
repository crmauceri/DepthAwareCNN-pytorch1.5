import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from depthaware.models.ops.depthconv.functional import DepthconvFunction

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
        return DepthconvFunction.apply(input, depth, self.weight, self.bias, self.stride,
                             self.padding, self.dilation)

    def output_size(self, input):
        channels = self.weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (self.weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


if __name__ == '__main__':
    import numpy as np

    batch_size = 4
    w, h = 5, 5
    kernel_size = 2
    out_channels = 1
    padding = 0
    dilation = 2
    stride = 1

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #Toy data
    input1 = torch.randn((batch_size, 3, w, h), requires_grad=True, device=device)
    input2 = input1.clone().detach().requires_grad_(True).to(device)  # Using True throws error on backward pass
    depth = torch.ones((batch_size, 1, w, h), device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    bias = True

    # Pytorch Conv2d pipeline
    conv = nn.Conv2d(3, out_channels, kernel_size, bias=True, padding=padding, dilation=dilation, stride=1)
    if torch.cuda.is_available():
        conv = conv.cuda()

    # DepthConv pipeline
    conv_test = DepthConv(3, out_channels, kernel_size, bias=True, padding=padding, dilation=dilation, stride=1)
    conv_size = conv_test.output_size(input2)

    conv_test.weight = nn.Parameter(
        conv.weight.clone().detach().requires_grad_(True))  # Copy weights and bias from conv so result should be same
    if bias:
        conv_test.bias = nn.Parameter(conv.bias.clone().detach().requires_grad_(True))

    if torch.cuda.is_available():
        conv_test = conv_test.cuda()

    #Shared layers
    fc = nn.Linear(torch.prod(torch.tensor(conv_size[1:])), 10)
    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        fc = fc.cuda()
        loss = loss.cuda()

    #Run forward pass
    conv_y = conv(input1)
    conv_loss = loss(fc(conv_y.view(-1, torch.prod(torch.tensor(conv_size[1:])))),
                     target)

    conv_test_y = conv_test(input2, depth)
    conv_test_loss = loss(fc(conv_test_y.view(-1, torch.prod(torch.tensor(conv_size[1:])))),
                          target)

    # The convolution forward results are equal within 5 decimal places
    np.testing.assert_array_almost_equal(conv_y.detach().cpu().numpy(), conv_test_y.detach().cpu().numpy(), decimal=5)

    # The gradient calculation is equal within 6 decimal places
    conv_loss.backward()
    conv_test_loss.backward()

    weight_grad = conv.weight.grad
    weight_grad_test = conv_test.weight.grad
    np.testing.assert_array_almost_equal(weight_grad.detach().cpu().numpy(), weight_grad_test.detach().cpu().numpy())

    if bias:
        bias_grad = conv.bias.grad
        bias_grad_test = conv_test.bias.grad
        np.testing.assert_array_almost_equal(bias_grad.detach().cpu().numpy(), bias_grad_test.detach().cpu().numpy())

    input_grad = input1.grad
    input_grad_test = input2.grad
    np.testing.assert_array_almost_equal(input_grad.detach().cpu().numpy(), input_grad_test.detach().cpu().numpy())
