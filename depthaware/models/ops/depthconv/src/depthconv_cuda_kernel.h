torch::Tensor depthconv_im2col(
    torch::Tensor data_im,
    torch::Tensor data_depth,
    const int channels, const int height, const int width,
    const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w);

torch::Tensor depthconv_col2im(
    torch::Tensor data_col,
    torch::Tensor data_depth,
    const int channels, const int height, const int width,
    const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w) ;


