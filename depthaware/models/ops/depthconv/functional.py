import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import depthconv


def depth_conv(input,
                  depth,
                  weight,
                  bias,
                  stride=1,
                  padding=0,
                  dilation=1):

    if input is not None and input.dim() != 4:
        raise ValueError(
            "Expected 4D tensor as input, got {}D tensor instead.".format(
                input.dim()))

    f = DepthconvFunction(
        _pair(stride), _pair(padding), _pair(dilation))
    # print bias
    if isinstance(bias, torch.nn.Parameter):
        return f(input, depth, weight, bias)
    else:
        return f(input, depth, weight)



class DepthconvFunction(Function):
    def __init__(self, stride, padding, dilation, bias=True):
        super(DepthconvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, input, depth, weight, bias):
        # print('forward')
        if bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device)

        self.save_for_backward(input, depth, weight, bias)

        output_size = [int((input.size()[i + 2] + 2 * self.padding[i] - weight.size()[i + 2]) / self.stride[i] + 1)
                       for i in range(2)]
        self.columns = input.new(weight.size(1) * weight.size(2) * weight.size(3),
                                  output_size[0] * output_size[1]).zero_()
        self.ones = input.new(output_size[0] * output_size[1]).zero_()

        if not input.is_cuda:
            raise NotImplementedError
        else:
            return depthconv.forward(
                    input, depth, weight, bias, self.columns, self.ones,
                    weight.size(3), weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1], self.dilation[0])

        return output

    def backward(self, grad_output):
        # print('backward')
        input, depth, weight, bias = self.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError
            if self.needs_input_grad[0]:
                grad_input = depthconv.backward_input(
                    input, depth, grad_output, weight, self.columns,
                    weight.size(3), weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1], self.dilation[0])

            if self.needs_input_grad[2]:
                grad_weight, grad_bias = depthconv.backward_parameters(
                    input, depth, grad_output, self.columns, self.ones,
                    weight.size(3), weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], 1)

                if len(self.needs_input_grad) == 4:
                    if not self.needs_input_grad[3]:
                        grad_bias = None
                else:
                    grad_bias = None

        return grad_input, None, grad_weight, grad_bias

    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size
