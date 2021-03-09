import torch
from torch.autograd import Function

if torch.cuda.is_available():
    import depthavgpooling

class DepthavgpoolingFunction(Function):
    @staticmethod
    def outputSize(input, kernel_size, stride, padding):
        width = int((input.size(-2) + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
        height = int((input.size(-1) + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)

        output_size = [input.size(0), kernel_size[0], width, height]
        # print(output_size)
        if not all([s > 0 for s in output_size]):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))

        return output_size

    @staticmethod
    def forward(ctx, input, depth, kernel_size=[3,3], alpha=1.0, stride=[1,1], padding=[0,0], useDepth=True):
        ctx.save_for_backward(input, depth)

        ctx.depthweightcount = input.new(*(depth.size())).zero_()
        ctx.stride = stride
        ctx.padding = padding
        ctx.kernel_size = kernel_size
        ctx.alpha = alpha
        ctx.useDepth = useDepth

        # print("AvgPooling: input: {}, kernel:{}, stride:{}, padding:{}".format(input.shape, ctx.kernel_size, ctx.stride, ctx.padding))

        if not input.is_cuda:
            raise NotImplementedError
        else:
            return depthavgpooling.forward(
                    input, depth, ctx.depthweightcount,
                    kernel_size[1], kernel_size[0], stride[1], stride[0],
                    padding[1], padding[0], useDepth)

    @staticmethod
    def backward(ctx, grad_output):
        input, depth = ctx.saved_tensors
        grad_input = None

        # print("AvgPooling Backward: kernel:{}, stride:{}, padding:{}".format(ctx.kernel_size, ctx.stride, ctx.padding))

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            try:
                grad_input = depthavgpooling.backward(
                    input, depth, ctx.depthweightcount, grad_output,
                    ctx.kernel_size[1], ctx.kernel_size[0], ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.useDepth)
            except RuntimeError as e:
                print("Error in AvgPooling: kernel:{}, stride:{}, padding:{}".format(ctx.kernel_size, ctx.stride, ctx.padding))
                raise e
        return grad_input, None, None, None, None, None, None
