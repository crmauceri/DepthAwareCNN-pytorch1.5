import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from depthaware.models.ops.depthavgpooling.functional import DepthavgpoolingFunction

class DepthAvgPooling(Module):
    def __init__(self,
                 kernel_size,
                 alpha=1,
                 stride=1,
                 padding=0):
        super(DepthAvgPooling, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.alpha = alpha
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, input, depth):
        return DepthavgpoolingFunction.apply(input, depth, self.kernel_size, self.alpha, self.stride, self.padding)

if __name__ == '__main__':
    import numpy as np

    batch_size = 5
    w, h = 7, 15
    kernel_size = 3
    out_channels = 2
    padding = 0
    dilation = 2
    stride = 1

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #Toy data
    input1 = torch.randn((batch_size, 3, w, h), requires_grad=True, device=device)
    input2 = input1.clone().detach().requires_grad_(True).to(device) # Using True throws error on backward pass
    depth = torch.ones((batch_size, 1, w, h), device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    bias = True

    # Pytorch AvgPool2d Pipeline
    pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
    if torch.cuda.is_available():
        pool = pool.cuda()
    pool_y = pool(input1)

    fc = nn.Linear(torch.prod(torch.tensor(pool_y.shape[1:])), 10)
    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        fc = fc.cuda()
        loss = loss.cuda()

    pool_loss = loss(fc(pool_y.view(-1, torch.prod(torch.tensor(pool_y.shape[1:])))),
                     target)

    # DepthAvgPool Pipeline
    pool_test = DepthAvgPooling(kernel_size, stride, padding)
    if torch.cuda.is_available():
        pool_test = pool_test.cuda()

    pool_test_y = pool_test(input2, depth)
    assert(pool_y.shape == pool_test_y.shape)

    pool_test_loss = loss(fc(pool_test_y.view(-1, torch.prod(torch.tensor(pool_y.shape[1:])))),
                     target)

    # The convolution forward results are equal within 6 decimal places
    np.testing.assert_array_almost_equal(pool_y.detach().cpu().numpy(), pool_test_y.detach().cpu().numpy())

    # The gradient calculations are equal within 6 decimal places
    pool_loss.backward()
    pool_test_loss.backward()

    input_grad = input1.grad
    input_grad_test = input2.grad
    np.testing.assert_array_almost_equal(input_grad.detach().cpu().numpy(), input_grad_test.detach().cpu().numpy())