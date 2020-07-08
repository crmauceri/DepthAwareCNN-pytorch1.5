import torch
import numpy as np
import unittest
import math
import time

from depthaware.models.ops.depthavgpooling.functional import DepthavgpoolingFunction

## Run unit tests with python -m unittest depthavgpooling_unit_tests.py

#Pass through loss for manually manipulating backprop gradients
class SimpleLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        return input - target

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def compareImplementations(input, depth, alpha, kernel_size,
                           stride, padding,
                           target, grad_output, useDepth=True):

    #Depth CNN Implementation
    start_forward = time.time()
    x_test = DepthavgpoolingFunction.apply(input, depth, kernel_size, alpha, stride, padding, useDepth)
    end_foward = time.time()
    loss = SimpleLoss.apply(x_test, target)

    start_backward = time.time()
    loss.backward(grad_output)
    end_backward = time.time()

    print("DepthAvgPool forward {}s, backward {}s".format(end_foward-start_forward, end_backward, start_backward))

    depth_input_grad = input.grad.cpu()

    pool_layer = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    #Pytorch CNN Implementation
    input = input.clone().detach().cuda().requires_grad_(True)
    start_forward = time.time()
    x = pool_layer(input)
    end_foward = time.time()
    target = target.clone().detach().cuda()
    loss = SimpleLoss.apply(x, target)
    start_backward = time.time()
    loss.backward(grad_output)
    end_backward = time.time()

    print("AvgPool2d forward {}s, backward {}s".format(end_foward - start_forward, end_backward, start_backward))

    # Check that the values are equal for the first 5 sig figs.
    pairs = [[input.grad.cpu().detach().numpy(), depth_input_grad.numpy()]]
    name = ['input grad']
    ret = []
    for i, pair in enumerate(pairs):
        #Find the order of magnitude
        mag = math.pow(10, math.floor(math.log10(pair[0].max())))
        left = pair[0]/mag
        right = pair[1]/mag
        ret.append({'var_name':name[i], 'tensors':(left, right)})
    return ret

def toy_data(batch_size, in_channels, w, h, kernel_size, stride, padding, device):
    input_size = (batch_size, in_channels, w, h)
    input = 0.01 * torch.tensor(range(input_size[0] * input_size[1] * input_size[2] * input_size[3]),
                                dtype=torch.float, device=device, requires_grad=True).reshape(input_size)
    input.retain_grad()

    depth = torch.ones((batch_size, 1, w, h), device=device)

    outsize = DepthavgpoolingFunction.outputSize(input, kernel_size, stride, padding)
    grad_output = torch.tensor(range(outsize[0] * outsize[1] * outsize[2] * outsize[3]),
                               dtype=torch.float, device=device).reshape(outsize)

    target = torch.zeros(outsize, device=device)

    return input, depth, grad_output, target

class DepthavgpoolingTests(unittest.TestCase):

    # Test vanilla configuration, depth is ones, no stride, no dilation
    def test_basic(self):
        batch_size = 5
        w, h = 9, 9
        kernel_size = (3, 3)
        stride = [1, 1]
        padding = [0, 0]
        alpha = 1.0
        device = torch.device('cuda')

        input, depth, grad_output, target = toy_data(batch_size, 3, w, h, kernel_size, stride, padding, device)

        result = compareImplementations(input, depth, alpha, kernel_size,
                           stride, padding,
                           target, grad_output)

        for pair in result:
            self.assertTrue(np.allclose(pair['tensors'][0], pair['tensors'][1]),
                        msg="Variable {} is not equal within 5 sig figs: {} \n {} ".format(pair['var_name'],
                                                                                           pair['tensors'][0],
                                                                                           pair['tensors'][1]))
    def test_padding(self):
        batch_size = 1
        w, h = 3, 3
        kernel_size = (3,3)
        stride = [1, 1]
        padding = [1, 1]
        alpha = 1.0
        device = torch.device('cuda')

        input, depth, grad_output, target = toy_data(batch_size, 3, w, h, kernel_size, stride, padding, device)

        result = compareImplementations(input, depth, alpha, kernel_size,
                                        stride, padding,
                                        target, grad_output, useDepth=False)

        for pair in result:
            self.assertTrue(np.allclose(pair['tensors'][0], pair['tensors'][1]),
                            msg="Variable {} is not equal within 5 sig figs: {} \n {} ".format(pair['var_name'],
                                                                                               pair['tensors'][0],
                                                                                               pair['tensors'][1]))


    def test_stride(self):
        batch_size = 1
        w, h = 5, 5
        kernel_size = (3,3)
        stride = [2, 2]
        padding = [0, 0]
        alpha = 1.0
        device = torch.device('cuda')

        input, depth, grad_output, target = toy_data(batch_size, 3, w, h, kernel_size, stride, padding, device)

        result = compareImplementations(input, depth, alpha, kernel_size,
                                        stride, padding,
                                        target, grad_output, useDepth=False)

        for pair in result:
            self.assertTrue(np.allclose(pair['tensors'][0], pair['tensors'][1]),
                            msg="Variable {} is not equal within 5 sig figs: {} \n {} ".format(pair['var_name'],
                                                                                               pair['tensors'][0],
                                                                                               pair['tensors'][1]))


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise ValueError('Need CUDA enabled pytorch')
    unittest.main()