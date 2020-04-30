import torch
import torch.nn as nn
import math


def depth_weights_F(alpha, depth_window):
    kernel_size = depth_window.shape[2]
    cur_depth_pix = depth_window[:, :, math.floor(kernel_size/2), math.floor(kernel_size/2)].unsqueeze(2).unsqueeze(3)
    return torch.exp(-alpha * torch.abs(cur_depth_pix.expand(depth_window.shape) - depth_window))


class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, alpha=8.3):
        super(DepthConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = torch.nn.Parameter(torch.randn((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn((1, self.out_channels)))
        else:
            self.bias = torch.nn.Parameter(torch.zeros((1, self.out_channels)))
        self.stride = stride
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        self.dilation = dilation
        self.alpha = alpha

    def forward(self, input, depth):
        output_size = self.output_size(input)

        input = self.pad(input)
        depth = self.pad(depth)
        b, c, w, h = input.shape

        output = torch.zeros(output_size)

        window_width = self.kernel_size*self.dilation
        calc_dim = (b, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        for i in range(0, output_size[2]):
            for j in range(0, output_size[3]):
                x = i*self.stride
                y = j*self.stride
                depth_window = depth[:, :, x:x+window_width:self.dilation, y:y+window_width:self.dilation]
                input_window = input[:, :,  x:x+window_width:self.dilation, y:y+window_width:self.dilation]
                depth_weights = depth_weights_F(self.alpha, depth_window)

                output[:, :, i, j] = \
                    torch.sum(
                        torch.sum(
                            torch.sum(
                                torch.mul(
                                    torch.mul(depth_weights.unsqueeze(1).expand(calc_dim), self.weight.unsqueeze(0).expand(calc_dim)),
                                    input_window.unsqueeze(1).expand(calc_dim)), dim=-1),
                            dim=-1),
                        dim=-1) \
                    + self.bias.expand((b, self.out_channels))

        return output

    def output_size(self, input):
        batch_n, channels, n_w, n_h = input.shape

        dilated_kernel = self.kernel_size + (self.kernel_size-1)*(self.dilation-1)

        w = math.floor((n_w + self.padding*2 - dilated_kernel + self.stride) / self.stride)
        h = math.floor((n_h + self.padding*2 - dilated_kernel + self.stride) / self.stride)
        return (batch_n, self.out_channels, w, h)


class DepthAvgPooling(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, alpha=8.3):
        super(DepthAvgPooling, self).__init__()

        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)

        self.alpha = alpha

    def forward(self, input, depth):
        input = self.pad(input)
        depth = self.pad(depth)

        output_size = self.output_size(input)
        b, c, w, h = input.shape

        output = torch.zeros(output_size)

        for i in range(0, output_size[2]):
            for j in range(0, output_size[3]):
                x = i*self.stride
                y = j*self.stride
                depth_window = depth[:, :, x:x+self.kernel_size, y:y+self.kernel_size]
                input_window = input[:, :, x:x+self.kernel_size, y:y+self.kernel_size]
                depth_weights = depth_weights_F(self.alpha, depth_window)
                depth_weight_total = torch.sum(torch.sum(torch.sum(depth_weights, dim=-1), dim=-1), dim=1)

                calc_dim = (b, c, self.kernel_size, self.kernel_size)
                output[:, :, i, j] = \
                    torch.div(
                            torch.sum(
                                torch.sum(
                                    torch.mul(depth_weights.expand(calc_dim), input_window), dim=-1),
                                dim=-1),
                        depth_weight_total.unsqueeze(-1).expand((b, c)))

        return output

    # TODO No dilation?
    def output_size(self, input_with_padding):
        batch_n, channels, n_w, n_h = input_with_padding.shape

        w = math.floor((n_w - self.kernel_size + self.stride) / self.stride)
        h = math.floor((n_h - self.kernel_size + self.stride) / self.stride)
        return (batch_n, channels, w, h)


if __name__ == '__main__':
    import numpy as np

    #Test that the result is the same as normal convolution when no depth
    input = torch.randn((2, 3, 7, 15))
    depth = torch.ones((2, 1, 7, 15))
    bias = False

    conv = DepthConv(3, 2, 3, padding=0,dilation=2)
    test_conv = conv(input, depth)

    torch_conv = torch.nn.functional.conv2d(input, conv.weight, bias=conv.bias.squeeze(), padding=0, dilation=2)
    np.testing.assert_array_almost_equal(torch_conv, test_conv, decimal=0.5)

    pool = DepthAvgPooling(3,padding=0)
    test_pool = pool(torch_conv, depth)
    torch_pool = nn.functional.avg_pool2d(torch_conv, 3, stride=1, padding=0)
    np.testing.assert_array_almost_equal(torch_pool, test_pool)

