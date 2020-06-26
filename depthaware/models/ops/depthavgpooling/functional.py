import torch
from torch.autograd import Function
import depthavgpooling

class DepthavgpoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, depth, kernel_size=[3,3], stride=[1,1], padding=[0,0]):
        ctx.save_for_backward(input, depth)

        ctx.depthweightcount = input.new(*(depth.size())).zero_()

        if not input.is_cuda:
            raise NotImplementedError
        else:
            return depthavgpooling.forward(
                    input, depth, ctx.depthweightcount,
                    kernel_size[1], kernel_size[0], stride[1], stride[0],
                    padding[1], padding[0])

    @staticmethod
    def backward(ctx, grad_output, kernel_size=[3,3], stride=[1,1], padding=[0,0]):
        input, depth = ctx.saved_tensors
        grad_input = None

        print("AvgPooling: kernel:{}, stride:{}, padding:{}".format(kernel_size, stride, padding))

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            try:
                grad_input = depthavgpooling.backward(
                    input, depth, ctx.depthweightcount, grad_output,
                    kernel_size[1], kernel_size[0], stride[1], stride[0],
                    padding[1], padding[0])
            except RuntimeError as e:
                print("Error in AvgPooling: kernel:{}, stride:{}, padding:{}".format(kernel_size, stride, padding))
                raise e
        return grad_input, None, None, None, None
