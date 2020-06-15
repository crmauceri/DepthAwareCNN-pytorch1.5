import torch
from torch.autograd import Function
import depthconv

class DepthconvFunction(Function):
    @staticmethod
    def outputSize(input, weight, stride, padding, dilation):
        weight_size = [(weight.size(i+2)-1)*(dilation[i]-1) + weight.size(i+2) for i in range(2)]
        width = ((input.size(-2) + 2 * padding[0] - weight_size[0]) / stride[0] + 1)
        height = ((input.size(-1) + 2 * padding[1] - weight_size[1]) / stride[1] + 1)

        output_size = [input.size(0), weight.size(0), width, height]
        print(output_size)
        if not all([s > 0 for s in output_size]):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))

        return output_size

    @staticmethod
    def forward(ctx, input, depth, weight, bias, alpha, stride, padding, dilation):
        # print('forward')
        if weight.size(2)% 2 == 0 or weight.size(2) % 2 == 0:
            raise ValueError("Function only defined for odd-sized kernels")
        output_size = DepthconvFunction.outputSize(input, weight, stride, padding, dilation)

        if bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device)

        # Crop to compatible size from input
        #TODO check for completeness
        #If stride is even, image width must be even
        if input.size(2) % 2 == 1 and stride[0] % 2 == 0:
            input = input[:,:,0:-2,:].contiguous()
            depth = depth[:, :, 0:-2, :].contiguous()
        #And vice-versa
        elif input.size(2) % 2 == 0 and stride[0] % 2 == 1:
            input = input[:, :, 0:-2, :].contiguous()
            depth = depth[:, :, 0:-2, :].contiguous()
        #Do the same for height
        if input.size(3) % 2 == 1 and stride[1] % 2 == 0:
            input = input[:,:,:,0:-2].contiguous()
            depth = depth[:, :, :, 0:-2].contiguous()
        elif input.size(3) % 2 == 0 and stride[1] % 2 == 1:
            input = input[:,:,:,0:-2].contiguous()
            depth = depth[:, :, :, 0:-2].contiguous()

        ctx.save_for_backward(input, depth, weight, bias)
        ctx.alpha = alpha
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

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
