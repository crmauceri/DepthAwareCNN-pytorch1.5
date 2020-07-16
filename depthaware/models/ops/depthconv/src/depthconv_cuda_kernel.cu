#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cstdio>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>


#define CUDA_KERNEL_LOOP(i, n)                                                 \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
    i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}

// Fills data_col((khxkwxC)x(HxW)) with the depth difference weighted image values
template <typename scalar_t>
__global__ void depthconv_im2col_gpu_kernel(
    const int n, const scalar_t* data_im, const scalar_t* data_depth, const int alpha,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    scalar_t* data_col) {

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

        //For each kernel element
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

                    // Weight image value by depth difference
                    val *= exp(-alpha * abs(Di - Dval));
                }
                *data_col_ptr = val;

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

torch::Tensor depthconv_im2col(
    torch::Tensor data_im,
    torch::Tensor data_depth,
    const double alpha, //Scaling factor
    const int channels, const int height, const int width,
    const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w) {

    throw_on_cuda_error( cudaPeekAtLastError(), "depthconv_cuda_kernel", 111 );
    throw_on_cuda_error( cudaDeviceSynchronize(), "depthconv_cuda_kernel", 112 );

    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;

    //std::cout << height << ", " << pad_h << ", " << dilation_h << ", " << ksize_h << ", " << stride_h << std::endl;
    //std::cout << width << ", " << pad_w << ", " << dilation_w << ", " << ksize_w << ", " << stride_w << std::endl;
    //std::cout << "Create column matrix: " << channels * ksize_h * ksize_w << "x" << height_col * width_col << std::endl;

    torch::Tensor data_col = torch::zeros({channels * ksize_h * ksize_w, height_col * width_col}, torch::kCUDA);

    // Launch
    AT_DISPATCH_FLOATING_TYPES(data_im.scalar_type(), "depthconv_im2col_gpu_kernel", ([&] {
        depthconv_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data_ptr<scalar_t>(), data_depth.data_ptr<scalar_t>(), alpha,
            height, width, channels, ksize_h, ksize_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
            data_col.data_ptr<scalar_t>() );
        }));

    throw_on_cuda_error( cudaPeekAtLastError(), "depthconv_cuda_kernel", 132 );
    throw_on_cuda_error( cudaDeviceSynchronize(), "depthconv_cuda_kernel", 133 );

    return data_col;
}

// Fills data_col((khxkw)x(CxHxW)) with the depth difference weighted image values
template <typename scalar_t>
__global__ void depthconv_grad2col_gpu_kernel(
    const int n, const scalar_t* data_im, const scalar_t* data_depth, const int alpha,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col, const int channels,
    scalar_t* data_col) {

    // CxHxW --> (khxkwxC)x(HxW)
    CUDA_KERNEL_LOOP(index, n) {
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col) / height_col;
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        scalar_t* data_col_ptr = data_col + w_col + h_col * width_col + c_im * width_col * height_col;
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

        //For each kernel element
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

                    // Weight image value by depth difference
                    val *= exp(-alpha * abs(Di - Dval));
                }
                *data_col_ptr = val;
                data_col_ptr += height_col * width_col * channels;
            }
        }
    }
}

torch::Tensor depthconv_gradOut2col(
    torch::Tensor data_im,
    torch::Tensor data_depth,
    const double alpha, //Scaling factor
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

    //std::cout << height << ", " << pad_h << ", " << dilation_h << ", " << ksize_h << ", " << stride_h << std::endl;
    //std::cout << width << ", " << pad_w << ", " << dilation_w << ", " << ksize_w << ", " << stride_w << std::endl;
    //std::cout << "Create column matrix: " << ksize_h * ksize_w << "x" << channels * height_col * width_col << std::endl;

    torch::Tensor data_col = torch::zeros({ksize_h * ksize_w, channels * height_col * width_col}, torch::kCUDA);

    // Launch
    AT_DISPATCH_FLOATING_TYPES(data_im.scalar_type(), "depthconv_grad2col_gpu_kernel", ([&] {
        depthconv_grad2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data_ptr<scalar_t>(), data_depth.data_ptr<scalar_t>(), alpha,
            height, width, ksize_h, ksize_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, channels,
            data_col.data_ptr<scalar_t>() );
        }));

    throw_on_cuda_error( cudaPeekAtLastError() , "depthconv_cuda_kernel", 235);
    throw_on_cuda_error( cudaDeviceSynchronize(), "depthconv_cuda_kernel", 236 );

    return data_col;
}
