#include <torch/extension.h>

#ifndef DEPTH_AVG_POOL_CUDA_KERNEL
#define DEPTH_AVG_POOL_CUDA_KERNEL

void AvePoolForward(const int count,
    torch::Tensor const input_data, torch::Tensor const input_depth_data, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    torch::Tensor const top_data, torch::Tensor const depth_weight_count);

void AvePoolBackward(const int count,
    torch::Tensor const gradOutput, torch::Tensor const input_depth,
    torch::Tensor const depth_weight_count,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    torch::Tensor const bottom_diff);

#endif