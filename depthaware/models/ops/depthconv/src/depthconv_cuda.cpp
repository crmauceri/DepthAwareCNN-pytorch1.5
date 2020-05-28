#include "depthconv_cuda_kernel.h"

#include <torch/extension.h>
#include <THC/THC.h>
#include <stdexcept>
#include <memory>
#include <string>

extern THCState *state;

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

void shape_check_forward(torch::Tensor input, torch::Tensor input_depth, torch::Tensor weight,
    int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW) {

    if(weight.ndimension() != 4){
        throw std::invalid_argument(string_format("4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
            "but got: %s", weight.ndimension()));
    }

    if(kW <= 0 || kH <= 0){
        throw std::invalid_argument(string_format("kernel size should be greater than zero, but got kH: %d kW: %d",
            kH, kW));
    }

    if(!(weight.size(2) == kH && weight.size(3) == kW)){
        throw std::invalid_argument(string_format("kernel size should be consistent with weight, but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
            kW, weight.size(2), weight.size(3)));
    }

    if(dW <= 0 || dH <= 0){
        throw std::invalid_argument(string_format("stride should be greater than zero, but got dH: %d dW: %d", dH, dW));
    }

    if(dilationW <= 0 || dilationH <= 0){
        throw std::invalid_argument(string_format("dilation should be greater than 0, but got dilationH: %d dilationW: %d",
            dilationH, dilationW));
    }

    //////////////////////////////////////////

    int ndim = input.ndimension();
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4) {
        dimf++;
        dimh++;
        dimw++;
    }

    if(ndim != 3 && ndim != 4){
        throw std::invalid_argument(string_format("3D or 4D input tensor expected but got: %s", ndim));
    }

    long nInputPlane = weight.size(1);
    long inputHeight = input.size(dimh);
    long inputWidth = input.size(dimw);
    long nOutputPlane = weight.size(0);

    long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

    if (outputWidth < 1 || outputHeight < 1){
        throw std::invalid_argument(string_format(
            "Given input size: (%ld x %ld x %ld). "
            "Calculated output size: (%ld x %ld x %ld). Output size is too small",
            nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
            outputWidth));
    }

    if(!(inputHeight >= kH && inputWidth >= kW)){
        throw std::invalid_argument("input image is smaller than kernel");
    }

/////////check depth map shape /////////

    int ndim_depth = input.ndimension();
    int dimf_depth = 0;
    int dimh_depth = 1;
    int dimw_depth = 2;

    if (ndim_depth == 4) {
        dimf_depth++;
        dimh_depth++;
        dimw_depth++;
    }

    if(ndim_depth != 3 && ndim_depth != 4){
        throw std::invalid_argument(string_format("3D input depth tensor expected but got: %s", ndim));
    }

    //long inputHeight_depth = input_depth->size[dimh_depth];
    //long inputWidth_depth = input_depth->size[dimw_depth];
    long inputHeight_depth = input_depth.size(dimh_depth);
    long inputWidth_depth = input_depth.size(dimw_depth);

    if(input_depth.size(1) != 1){
        throw std::invalid_argument("input depth should have only 1 channel");
    }

    if(!(inputHeight == inputHeight_depth && inputWidth == inputWidth_depth)){
        throw std::invalid_argument("input image and input depth should be the same size");
    }
}

void shape_check_bias(torch::Tensor weight, torch::Tensor bias){
    //////////// check bias //////////////////
    if(bias.ndimension() != 1){
        throw std::invalid_argument(string_format("Need bias of dimension %d but got %d", 1, bias.ndimension()));
    }

    if(bias.size(0) != weight.size(0)){
        throw std::invalid_argument(string_format("Need bias of size %d but got %d",
            weight.size(0), bias.size(0)));
    }
}

void shape_check_gradOutput(torch::Tensor input, torch::Tensor weight, torch::Tensor gradOutput,
    int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW){

    int ndim = input.ndimension();
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4) {
        dimf++;
        dimh++;
        dimw++;
    }

    long inputHeight = input.size(dimh);
    long inputWidth = input.size(dimw);
    long nOutputPlane = weight.size(0);

    long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

//////////////////////////////////////////
    if(gradOutput.size(dimf) != nOutputPlane){
        throw std::invalid_argument(string_format("invalid number of gradOutput planes, expected: %d, but got: %d",
            nOutputPlane, gradOutput.size(dimf)));
    }

    if(!(gradOutput.size(dimh) == outputHeight && gradOutput.size(dimw) == outputWidth)){
        throw std::invalid_argument(string_format("invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d",
            outputHeight, outputWidth, gradOutput.size(dimh), gradOutput.size(dimw)));
    }
}


torch::Tensor depthconv_forward_cuda(torch::Tensor input, torch::Tensor input_depth,
                             torch::Tensor weight, torch::Tensor bias,
                             int kW, int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW) {

    CHECK_INPUT(input);
    CHECK_INPUT(input_depth);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    shape_check_forward(input, input_depth, weight, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW);
    shape_check_bias(weight, bias);

    int batch = 1;
    if (input.ndimension() == 3) {
        // Force batch
        batch = 0;
        input = input.reshape({1, input.size(0), input.size(1), input.size(2)});
        input_depth = input_depth.reshape({1, input_depth.size(0), input_depth.size(1), input_depth.size(2)});
    }

    int batchSize = input.size(0);
    int nInputPlane = input.size(1);
    int inputHeight = input.size(2);
    int inputWidth = input.size(3);

    int nOutputPlane = weight.size(0);

    int outputWidth =
        (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    int outputHeight =
        (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    torch::Tensor output = torch::zeros({batchSize, nOutputPlane, outputHeight, outputWidth}, torch::kCUDA);
    torch::Tensor columns = torch::zeros({nInputPlane * kW * kH, outputHeight * outputWidth}, torch::kCUDA);

    torch::Tensor input_n;
    torch::Tensor depth_n;
    torch::Tensor output_n;

    //Repeat bias to match output size
    //Without the extra singleton dimensions the repeat function has the wrong dimensionality
    //bias = bias.reshape({bias.size(0), 1}).repeat({1, outputHeight*outputWidth});
    output = bias.reshape({bias.size(0), 1}).repeat({batchSize, 1, outputHeight, outputWidth});

    for (int elt = 0; elt < batchSize; elt++) {

        input_n = input.select(0, elt);
        depth_n = input_depth.select(0, elt);
        output_n = output.select(0, elt);

        //Reshape input and weight with depth difference
        columns = depthconv_im2col(input_n, depth_n,
            nInputPlane, inputHeight, inputWidth,
            kH, kW,
            padH, padW,
            dH, dW,
            dilationH, dilationW);

        torch::Tensor weight_slice = weight.view({weight.size(0), weight.size(1)*weight.size(2)*weight.size(3)});
        torch::Tensor output_slice = output_n.view({nOutputPlane, outputWidth*outputHeight});

        //Multiplication with reshaped input is equivalent to 2d convolution
        {
        using namespace torch::indexing;
        //output_slice.index_put_({Ellipsis}, torch::addmm(bias, weight_slice, columns));
        output_slice.addmm_(weight_slice, columns);
        }

       //Original code for reference
//        long m = weight.size(0);
//        long n = columns.size(1);
//        long k = input.size(1) * kH * kW;
//
//        THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
//                     columns.data(), n,
//                     weight.data(), k, 1.0f,
//                     output_n.data(), n);
    }

    if (batch == 0) {
        output = output.view({nOutputPlane, outputHeight, outputWidth});
    }

    return output;
}


std::vector<torch::Tensor> depthconv_backward_cuda(
    torch::Tensor input, torch::Tensor input_depth, torch::Tensor gradOutput,
    torch::Tensor weight, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW, double scale) {

    CHECK_INPUT(input);
    CHECK_INPUT(input_depth);
    CHECK_INPUT(gradOutput);
    CHECK_INPUT(weight);

    shape_check_forward(input, input_depth, weight, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW);
    shape_check_gradOutput(input, weight, gradOutput, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW);

    int batch = 1;
    if (input.ndimension() == 3) {
        // Force batch
        batch = 0;
        input = input.view({1, input.size(0), input.size(1), input.size(2)});
        gradOutput = gradOutput.view({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
    }

    int batchSize = input.size(0);
    int nInputPlane = input.size(1);
    int inputHeight = input.size(2);
    int inputWidth = input.size(3);

    int nOutputPlane = weight.size(0);

    int outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    int outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    if(input_depth.size(0) != batchSize){
        throw std::invalid_argument("invalid batch size of input depth");
    }

    //TODO Check these dimensions
    torch::Tensor gradWeight = torch::zeros_like(weight, torch::kCUDA);
    torch::Tensor gradBias = torch::zeros({nOutputPlane, 1}, torch::kCUDA);
    torch::Tensor gradInput = torch::zeros_like(input, torch::kCUDA);
    torch::Tensor ones = torch::ones({nOutputPlane, outputWidth*outputHeight}, torch::kCUDA);

    for (int elt = 0; elt < batchSize; elt++) {
        torch::Tensor input_depth_n = input_depth.select(0, elt);
        torch::Tensor gradOutput_n = gradOutput.select(0, elt);

        torch::Tensor gradOutput_n_slice = gradOutput_n.reshape({nOutputPlane, outputWidth*outputHeight}).transpose(1,0);
        torch::Tensor weight_slice = weight.view({nOutputPlane, weight.size(1)*weight.size(2)*weight.size(3)});
        torch::Tensor columns = torch::matmul(gradOutput_n_slice, weight_slice);

        //Original code for reference
//        long m = input.size(1) * kW * kH;
//        long n = columns.size(1);
//        long k = weight.size(0);
//
//        THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
//                     gradOutput_n.data(), n,
//                     weight.data(), m, 0.0f,
//                     columns.data(), n);

        torch::Tensor gradInput_n = depthconv_col2im(columns, input_depth_n,
            nInputPlane, inputHeight, inputWidth,
            kH, kW,
            padH, padW,
            dH, dW,
            dilationH, dilationW);

        {
        using namespace torch::indexing;
        gradInput.index_put_({elt, Ellipsis}, gradInput_n);
        }

        torch::Tensor gradWeight_slice = weight.view({nOutputPlane, weight.size(1)*weight.size(2)*weight.size(3)});
        {
        using namespace torch::indexing;
        //gradWeight.index_put_({Ellipsis}, gradWeight_slice.reshape({nOutputPlane, weight.size(1), weight.size(2), weight.size(3)}));
        gradWeight.index_put_({Ellipsis}, gradWeight_slice.addmm(columns.transpose(1,0), gradOutput_n_slice, /*beta=*/1.0, /*alpha=*/scale))
        }

        //Original code for reference
//        long m = nOutputPlane;
//        long n = nInputPlane * kW * kH;
//        long k = columns->size[1];
//
//        THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
//                     THCudaTensor_data(state, columns), k,
//                     THCudaTensor_data(state, gradOutput_n), k, 1.0f,
//                     THCudaTensor_data(state, gradWeight), n);

        std::cout << gradWeight << std::endl;

        // Do Bias:
        // Correct result!
        gradBias.addmm_(ones, gradOutput_n_slice, /*beta=*/1.0, /*alpha=*/scale);
    }

    if (batch == 0) {
        gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
        input = input.view({nInputPlane, inputHeight, inputWidth});
        input_depth = input_depth.view({1, inputHeight, inputWidth});
        gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
    }

    return {gradInput, gradWeight, gradBias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &depthconv_forward_cuda, "Depth Aware Convolution forward (CUDA)");
  m.def("backward", &depthconv_backward_cuda, "Depth Aware Convolution backward pass for input (CUDA)");
}