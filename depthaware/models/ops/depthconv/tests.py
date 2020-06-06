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

if __name__ == '__main__':
    import torch
    import numpy as np
    import depthconv

    #TODO check dilation and stride
    batch_size = 1
    w, h = 7, 7
    kernel_size = 3
    out_channels = 1
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input = torch.ones((batch_size, 3, w, h), device=device)
    depth = torch.ones((batch_size, 1, w, h), device=device)
    weight = torch.ones((out_channels, 3, kernel_size, kernel_size), device=device)
    outsize = output_size(input, weight, padding, dilation, stride)
    grad_output = torch.FloatTensor(range(outsize[0]*outsize[1]*outsize[2]*outsize[3])).cuda().reshape(outsize)

    print(grad_output)

    grad_input, grad_weight, grad_bias = depthconv.backward(
        input, depth, grad_output, weight, 1.0,
        weight.size(3), weight.size(2), stride[1], stride[0],
        padding[1], padding[0], dilation[1], dilation[0], 1.0)