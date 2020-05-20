#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cstdio>

#include "depthconv_cuda_kernel.h"

#define CUDA_KERNEL_LOOP(i, n)                                                 \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
    i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__global__ void depthconv_im2col_gpu_kernel(
    const int n, const scalar_t* data_im, const scalar_t* data_depth,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int height_col,
    const int width_col, scalar_t* data_col) {

    // CxHxW --> (khxkw)x(CxHxW)
    CUDA_KERNEL_LOOP(index, n) {
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col) / height_col;
        const int c_col = c_im * kernel_h * kernel_w;


        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        scalar_t* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        const scalar_t* data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
        const scalar_t* data_depth_ptr = data_depth + h_in * width + w_in;
        scalar_t Di = 0.;
        bool valid = true;
        if ((h_in + dilation_h * (kernel_h - 1) / 2)>=0 &&
            w_in  + dilation_w * (kernel_w - 1) / 2 >= 0 &&
            (h_in + dilation_h * (kernel_h - 1) / 2) < height &&
            w_in  + dilation_w * (kernel_w - 1) / 2 < width)

            Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2];
        else
            valid = false;
        //const scalar_t Di = data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + (w_in + (kernel_w - 1) / 2 + dilation_w - 1)];

        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                scalar_t val = static_cast<scalar_t>(0);
                scalar_t Dval = static_cast<scalar_t>(0);
                const int h_im = h_in + i * dilation_h;
                const int w_im = w_in + j * dilation_w;

                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                    const int map_h = i * dilation_h;
                    const int map_w = j * dilation_w;
                    val = data_im_ptr[map_h * width + map_w];

                    if (valid)
                        Dval = data_depth_ptr[map_h * width + map_w];

                    //printf("%f,%d\n",Dval,h_in * width + w_in+map_h * width + map_w - ((h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + (w_in + (kernel_w - 1) / 2 + dilation_w - 1)));
                    // printf("Di-Dval: %f, %f\n", Di, Dval);
                    // if (exp(-abs(Di - Dval))<0.2)
                    //	printf("Di-Dval: %f\n", exp(-abs(Di - Dval)));
                    val *= exp(-abs(Di - Dval));
                }
                *data_col_ptr = val;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

template <typename DType>
torch::Tensor depthconv_im2col(
    torch::Tensor data_im,
    torch::Tensor data_depth,
    const int channels, const int height, const int width,
    const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w) {

    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;

    torch::Tensor data_col = torch::zeros({channels * ksize_h * ksize_w, height_col * width_col});

    // Launch
    AT_DISPATCH_FLOATING_TYPES(data_im.type(), "depthconv_im2col_gpu_kernel", ([&] {
        depthconv_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data<scalar_t>(), data_depth.data<scalar_t>(),
            height, width, ksize_h, ksize_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col.data<scalar_t>());
        }));

    return data_col
}

template <typename scalar_t>
__global__ void depthconv_col2im_gpu_kernel(
    const int n,
    const scalar_t* data_col,
    const scalar_t* data_depth,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, const int height_col,
    const int width_col, scalar_t* grad_im) {

    CUDA_KERNEL_LOOP(index, n) {
        for (int ii = 0; ii < kernel_h * kernel_w; ii++){
            int ii_index = ii + index * kernel_h * kernel_w;
            const int j = (ii_index / width_col / height_col) % kernel_w;
            const int i = (ii_index / width_col / height_col / kernel_w) % kernel_h;
            const int c = ii_index / width_col / height_col / kernel_w / kernel_h;
            // compute the start and end of the output

            int w_out = ii_index % width_col;
            int h_out = (ii_index / width_col) % height_col;
            int w_in = w_out * stride_w - pad_w;
            int h_in = h_out * stride_h - pad_h;

            //const scalar_t cur_inv_h_data = h_in + i * dilation_h;
            //const scalar_t cur_inv_w_data = w_in + j * dilation_w;

            const scalar_t cur_top_grad = data_col[ii_index];
            const int cur_h = h_in + i * dilation_h;//(int)cur_inv_h_data;
            const int cur_w = w_in + j * dilation_w;//(int)cur_inv_w_data;

            scalar_t Di = 0.;
            bool valid = true;
            if ((h_in + dilation_h * (kernel_h - 1) / 2)>=0 &&
                w_in  + dilation_w * (kernel_w - 1) / 2 >= 0 &&
                (h_in + dilation_h * (kernel_h - 1) / 2) < height &&
                w_in  + dilation_w * (kernel_w - 1) / 2 < width)

                Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2];
            else
                valid = false;

            //const scalar_t Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2];
            //const scalar_t Di = data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + w_in  + (kernel_w - 1) / 2 + dilation_w - 1];
            //printf("%d\n",(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2);
            //data_depth[cur_h * width + cur_w];
            // data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + w_in  + (kernel_w - 1) / 2 + dilation_w - 1];

            int cur_bottom_grad_pos = (c * height + cur_h) * width + cur_w;
            int cur_bottom_depth_pos= (cur_h) * width + cur_w;

            //printf("%d,%d,%d,%d\n",i,j,((h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2-cur_bottom_depth_pos),dilation_h);
            //printf("%d\n",((h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2-cur_bottom_depth_pos));

            scalar_t Dval = 0.;
            if (valid)
                Dval = data_depth[cur_bottom_depth_pos];

            if (cur_h >= 0 && cur_h < height && cur_w  >= 0 && cur_w  < width)
                atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad * exp(-abs(Di - Dval)));

        }
    }
}

template <typename DType>
torch::Tensor depthconv_col2im(
    torch::Tensor data_col,
    torch::Tensor data_depth,
    const int channels, const int height, const int width,
    const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w) {

    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;

    torch::Tensor grad_im = torch::zeros({channels, height, width});

    // int channel_per_depthconv_group = channels / depthconv_group;
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    AT_DISPATCH_FLOATING_TYPES(data_col.type(), "depthconv_col2im", ([&] {
        depthconv_col2im_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            data_col.data<scalar_t>(),
            data_depth.data<scalar_t>(),
            channels, height, width,
            ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            height_col, width_col,
            grad_im.data<scalar_t>());
    }));

    return grad_im;
}
