import torch
import numpy as np
import unittest
import math

from depthaware.models.ops.depthconv.functional import DepthconvFunction

## Run unit tests with python -m unittest depthconv_unit_tests.py

#Pass through loss for manually manipulating backprop gradients
class SimpleLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        return input - target

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def compareImplementations(input, depth, weight, bias, alpha,
                           stride, padding, dilation,
                           target, grad_output, useDepth=True):

    #Depth CNN Implementation
    x_test = DepthconvFunction.apply(input, depth, weight, bias, alpha, stride, padding, dilation, useDepth)
    loss = SimpleLoss.apply(x_test, target)
    loss.backward(grad_output)

    depth_input_grad = input.grad.cpu()

    conv_layer = torch.nn.Conv2d(weight.size(1), weight.size(0), (weight.size(2), weight.size(3)),
                                 bias=True, stride=stride, padding=padding,
                                 dilation=dilation, groups=1)
    conv_layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=True)
    conv_layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=True)

    #Pytorch CNN Implementation
    input = input.clone().detach().cuda().requires_grad_(True)
    x = conv_layer(input)
    target = target.clone().detach().cuda()
    loss = SimpleLoss.apply(x, target)
    loss.backward(grad_output)

    # Check that the values are equal for the first 5 sig figs.
    pairs = [[input.grad.cpu().detach().numpy(), depth_input_grad.numpy()],
             [conv_layer.weight.grad.cpu().detach().numpy(), weight.grad.cpu().detach().numpy()],
             [conv_layer.bias.grad.cpu().detach().numpy(), bias.grad.cpu().detach().numpy()]]
    name = ['input grad', 'weight grad', 'bias grad']
    ret = []
    for i, pair in enumerate(pairs):
        #Find the order of magnitude
        mag = math.pow(10, math.floor(math.log10(pair[0].max())))
        left = pair[0]/mag
        right = pair[1]/mag
        ret.append({'var_name':name[i], 'tensors':(left, right)})
    return ret

def toy_data(batch_size, in_channels, w, h, out_channels, kernel_size, stride, padding, dilation, device):
    input_size = (batch_size, in_channels, w, h)
    input = 0.01 * torch.tensor(range(input_size[0] * input_size[1] * input_size[2] * input_size[3]),
                                dtype=torch.float, device=device, requires_grad=True).reshape(input_size)
    input.retain_grad()

    depth = torch.ones((batch_size, 1, w, h), device=device)
    weight_size = (out_channels, in_channels, kernel_size, kernel_size)
    weight = 0.01 * torch.tensor(range(weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]),
                                 dtype=torch.float, device=device, requires_grad=True).reshape(weight_size)
    weight.retain_grad()

    bias = torch.ones((out_channels), device=device, requires_grad=True)
    bias.retain_grad()

    outsize = DepthconvFunction.outputSize(input, weight, stride, padding, dilation)
    grad_output = torch.tensor(range(outsize[0] * outsize[1] * outsize[2] * outsize[3]),
                               dtype=torch.float, device=device).reshape(outsize)

    target = torch.zeros(outsize, device=device)

    return input, depth, weight, bias, grad_output, target

class DepthConvTests(unittest.TestCase):

    # Test vanilla configuration, depth is ones, no stride, no dilation
    def test_basic(self):
        batch_size = 5
        w, h = 9, 9
        kernel_size = 3
        out_channels = 2
        stride = [1, 1]
        padding = [0, 0]
        dilation = [1, 1]
        alpha = 1.0
        device = torch.device('cuda')

        input, depth, weight, bias, grad_output, target = toy_data(batch_size, 3, w, h, out_channels, kernel_size, stride, padding, dilation, device)

        result = compareImplementations(input, depth, weight, bias, alpha,
                           stride, padding, dilation,
                           target, grad_output)

        for pair in result:
            self.assertTrue(np.allclose(pair['tensors'][0], pair['tensors'][1]),
                        msg="Variable {} is not equal within 5 sig figs: {} \n {} ".format(pair['var_name'],
                                                                                           pair['tensors'][0],
                                                                                           pair['tensors'][1]))
    def test_dimentions_VGG(self):
        batch_size = 1
        kernel_size = 3
        stride = [1, 1]
        alpha = 1.0
        device = torch.device('cuda')

        width = [1000, 206, 162, 200, 178]
        channels = [3, 64, 128,  512, 512, 1024]
        padding = [(1,1), (1,1), (1,1), (2,2), (12,12)]
        dilation = [(1, 1), (1, 1), (1, 1), (2, 2), (12, 12)]

        error = False
        msg_list = []

        for i in range(5):
            w = width[i]
            h = w
            out_c= channels[i+1]
            in_c= channels[i]
            p = padding[i]
            d = dilation[i]

            input, depth, weight, bias, grad_output, target = toy_data(batch_size, in_c, w, h, out_c, kernel_size,
                                                                       stride, p, d, device)

            result = compareImplementations(input, depth, weight, bias, alpha,
                                            stride, p, d,
                                            target, grad_output, useDepth=False)


            for pair in result:
                if(not np.allclose(pair['tensors'][0], pair['tensors'][1])):
                    error = True
                    msg_list.append("VGG Layer {} {}: input:{}, depth:{}, kernel:{}, stride:{}, padding:{}, dilation:{}".format(i, pair['var_name'],
                                                                                    input.shape, depth.shape, weight.shape, stride, p, d))
                    print("Failed VGG Layer {} {} {}x{}".format(i, pair['var_name'], w, h))
                    print(pair['tensors'][0][0,0,:,:])
                    print(pair['tensors'][1][0,0,:,:])
                else:
                    print("Passed VGG Layer {} {} {}x{}".format(i, pair['var_name'], w, h))
        self.assertFalse(error, "\n".join(msg_list))


    def test_VGG(self):
        batch_size = 1
        w = 1065
        h = 2137
        device = torch.device('cuda')

        import depthaware.models.VGG_Deeplab as VGG_Deeplab
        model = VGG_Deeplab.vgg16(num_classes=1024,depthconv=True)
        model.cuda()

        criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()

        input_size = (batch_size, 3, w, h)
        input = 0.01 * torch.tensor(range(input_size[0] * input_size[1] * input_size[2] * input_size[3]),
                                    dtype=torch.float, device=device, requires_grad=True).reshape(input_size)
        input.retain_grad()

        depth = torch.ones((batch_size, 1, w, h), device=device)

        target = torch.ones((batch_size, 134, 268), device=device, dtype=torch.long)

        pred = model(input, depth)
        loss = criterionSeg(pred, target)

        loss.backward()
        self.assertTrue(True, msg="This should be impossible to trigger")


    def test_padding(self):
        batch_size = 1
        w, h = 3, 3
        kernel_size = 3
        out_channels = 2
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        alpha = 1.0
        device = torch.device('cuda')

        input, depth, weight, bias, grad_output, target = toy_data(batch_size, 3, w, h, out_channels, kernel_size,
                                                                   stride, padding, dilation, device)

        result = compareImplementations(input, depth, weight, bias, alpha,
                                        stride, padding, dilation,
                                        target, grad_output, useDepth=False)

        for pair in result:
            self.assertTrue(np.allclose(pair['tensors'][0], pair['tensors'][1]),
                            msg="Variable {} is not equal within 5 sig figs: {} \n {} ".format(pair['var_name'],
                                                                                               pair['tensors'][0],
                                                                                               pair['tensors'][1]))

    def test_dilation(self):
        batch_size = 1
        w, h = 6, 6
        kernel_size = 3
        out_channels = 2
        stride = [1, 1]
        padding = [0, 0]
        dilation = [2, 2]
        alpha = 1.0
        device = torch.device('cuda')

        input, depth, weight, bias, grad_output, target = toy_data(batch_size, 3, w, h, out_channels, kernel_size,
                                                                   stride, padding, dilation, device)

        result = compareImplementations(input, depth, weight, bias, alpha,
                                        stride, padding, dilation,
                                        target, grad_output, useDepth=False)

        for pair in result:
            self.assertTrue(np.allclose(pair['tensors'][0], pair['tensors'][1]),
                            msg="Variable {} is not equal within 5 sig figs: {} \n {} ".format(pair['var_name'],
                                                                                               pair['tensors'][0],
                                                                                               pair['tensors'][1]))

    def test_stride(self):
        batch_size = 1
        w, h = 5, 5
        kernel_size = 3
        out_channels = 2
        stride = [2, 2]
        padding = [0, 0]
        dilation = [1, 1]
        alpha = 1.0
        device = torch.device('cuda')

        input, depth, weight, bias, grad_output, target = toy_data(batch_size, 3, w, h, out_channels, kernel_size,
                                                                   stride, padding, dilation, device)

        result = compareImplementations(input, depth, weight, bias, alpha,
                                        stride, padding, dilation,
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