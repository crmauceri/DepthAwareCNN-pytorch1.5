import torch
import numpy as np
import depthconv

def output_size(input, weight, padding, dilation, stride):
    channels = weight.size(0)

    output_size = (input.size(0), channels)
    for d in range(input.dim() - 2):
        in_size = input.size(d + 2)
        pad = padding[d]
        kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
        stride_x = stride[d]
        output_size += ((in_size + (2 * pad) - kernel) // stride_x + 1,)
    if not all(map(lambda s: s > 0, output_size)):
        raise ValueError(
            "convolution input is too small (output would be {})".format(
                'x'.join(map(str, output_size))))
    return output_size

class TestLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        return input - target

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


if __name__ == '__main__':

    #TODO check dilation and stride
    batch_size = 1
    w, h = 5, 5
    kernel_size = 3
    out_channels = 1
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input = torch.randn((batch_size, 3, w, h), device=device)
    depth = torch.ones((batch_size, 1, w, h), device=device)
    weight = 0.5 * torch.ones((out_channels, 3, kernel_size, kernel_size), device=device)
    outsize = output_size(input, weight, padding, dilation, stride)
    grad_output = torch.FloatTensor(range(outsize[0]*outsize[1]*outsize[2]*outsize[3])).cuda().reshape(outsize)

    print("Toy grad output:")
    print(grad_output)

    grad_input, grad_weight, grad_bias = depthconv.backward(
        input, depth, grad_output, weight, 1.0,
        weight.size(3), weight.size(2), stride[1], stride[0],
        padding[1], padding[0], dilation[1], dilation[0], 1.0)

    print("DepthConv input gradient:")
    print(grad_input)

    conv_layer = torch.nn.Conv2d(out_channels, kernel_size, kernel_size, bias=True, stride=stride, padding=padding,
                                 dilation=dilation, groups=1)
    conv_layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=True)
    bias = torch.zeros((out_channels, 1), device=device)
    conv_layer.bias = torch.nn.Parameter(bias.squeeze(1), requires_grad=True)

    input = input.clone().detach().cuda().requires_grad_(True)
    x = conv_layer(input)
    target = torch.zeros(x.shape, device=device)
    loss = TestLoss.apply(x, target)
    loss.backward(grad_output)

    print("Pytorch input gradient:")
    print(input.grad.cpu())

    print("DepthConv weight gradient:")
    print(grad_weight)

    print("Pytorch weight gradient:")
    print(conv_layer.weight.grad.cpu())

    print("DepthConv bias gradient:")
    print(grad_bias)

    print("Pytorch bias gradient:")
    print(conv_layer.bias.grad.cpu())

    np.testing.assert_array_almost_equal(grad_input.cpu().detach().numpy(), input.grad.cpu().detach().numpy())
    np.testing.assert_array_almost_equal(grad_weight.cpu().detach().numpy(), conv_layer.weight.grad.cpu().detach().numpy())
    np.testing.assert_array_almost_equal(grad_bias.cpu().detach().numpy(), conv_layer.bias.grad.cpu().detach().numpy())