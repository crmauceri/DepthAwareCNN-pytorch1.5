import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import depthconv

class DepthconvFunction(Function):
    @staticmethod
    def forward(ctx, input, depth, weight, bias, alpha, stride, padding, dilation):
        # print('forward')
        if weight.size(2)% 2 == 0 or weight.size(2) % 2 == 0:
            raise ValueError("Function only defined for odd-sized kernels")

        if bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device)

        # Crop to compatible size from input
        #TODO check for completeness
        #If stride is even, image width must be even
        if input.size(2) % 2 == 1 and stride[0] % 2 == 0:
            input = input[:,:,0:-2,:]
        #And vice-versa
        elif input.size(2) % 2 == 0 and stride[0] % 2 == 1:
            input = input[:, :, 0:-2, :]
        #Do the same for height
        if input.size(3) % 2 == 1 and stride[1] % 2 == 0:
            input = input[:,:,:,0:-2]
        elif input.size(3) % 2 == 0 and stride[1] % 2 == 1:
            input = input[:,:,:,0:-2]

        ctx.save_for_backward(input, depth, weight, bias)
        ctx.alpha = alpha
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        output_size = [int((input.size()[i + 2] + 2 * padding[i] - weight.size()[i + 2]) / stride[i] + 1)
                       for i in range(2)]




        if not input.is_cuda:
            raise NotImplementedError
        else:
            return depthconv.forward(
                    input, depth, weight, bias, alpha,
                    weight.size(3), weight.size(2), stride[1], stride[0],
                    padding[1], padding[0], dilation[1], dilation[0])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        input, depth, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError

            grad_input, grad_weight, grad_bias = depthconv.backward(
                input, depth, grad_output, weight, ctx.alpha,
                weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], 1.0)

        return grad_input, None, grad_weight, grad_bias, None, None, None, None
