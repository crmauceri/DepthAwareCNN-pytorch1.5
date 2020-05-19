
torch::Tensor depthavgpooling_forward_cuda(
    torch::Tensor input,
    torch::Tensor input_depth,
    torch::Tensor depthweightcount,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH);

torch::Tensor depthavgpooling_backward_cuda(
    torch::Tensor input,
    torch::Tensor input_depth,
    torch::Tensor depthweightcount,
    torch::Tensor gradOutput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH);
