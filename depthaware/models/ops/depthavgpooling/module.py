import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from depthaware.models.ops.depthavgpooling.functional import depth_avgpooling

class Depthavgpooling(Module):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(Depthavgpooling, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, input, depth):
        return depth_avgpooling(input, depth, self.kernel_size, self.stride, self.padding)

if __name__ == '__main__':
    import numpy as np

    batch_size = 5
    w, h = 7, 15
    kernel_size = 3
    out_channels = 2
    padding = 0
    dilation = 2
    stride = 1

    #Toy data
    input1 = torch.randn((batch_size, 3, w, h), requires_grad=True)
    input2 = input1.clone().detach().requires_grad_(True) # Using True throws error on backward pass
    depth = torch.ones((batch_size, 1, w, h))
    target = torch.randint(0, 10, (batch_size,))
    bias = True

    # Pytorch AvgPool2d Pipeline
    pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
    pool_y = pool(input1)

    fc = nn.Linear(torch.prod(torch.tensor(pool_y.shape[1:])), 10)
    loss = nn.CrossEntropyLoss()

    pool_y = pool(input1)
    pool_loss = loss(fc(pool_y.view(-1, torch.prod(torch.tensor(pool_y.shape[1:])))),
                     target)

    # DepthAvgPool Pipeline
    pool_test = Depthavgpooling(kernel_size, stride, padding)
    pool_test_y = pool_test((input2, depth))

    assert(pool_y.shape == pool_test_y.shape)

    pool_test_loss = loss(fc(pool_test_y.view(-1, torch.prod(torch.tensor(pool_y.shape[1:])))),
                     target)

    # The convolution forward results are equal within 6 decimal places
    np.testing.assert_array_almost_equal(pool_y.detach().numpy(), pool_test_y.detach().numpy())

    # The gradient calculations are equal within 6 decimal places
    pool_loss.backward()
    pool_test_loss.backward()

    input_grad = input1.grad
    input_grad_test = input2.grad
    np.testing.assert_array_almost_equal(input_grad.detach().numpy(), input_grad_test.detach().numpy())