#include "depthavgpooling_cuda_kernel.h"

#include <torch/extension.h>
#include <stdexcept>
#include <memory>
#include <string>


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "depthavgpool: Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

void shape_check_forward(
    torch::Tensor input, torch::Tensor input_depth,
    int kH, int kW, int dH, int dW, int padH, int padW) {

    if (kW <= 0 || kH <= 0 ) {
        throw std::invalid_argument(string_format("depthavgpool: kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW));
    }

    if (dW <= 0 || dH <= 0) {
        throw std::invalid_argument(string_format("depthavgpool: stride should be greater than zero, but got dH: %d dW: %d", dH, dW));
    }

    int ndim = input.ndimension();
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
    }

    if (ndim != 3 && ndim != 4){
        throw std::invalid_argument("depthavgpool: 3D or 4D input tensor expected but got: " + std::to_string(ndim));
    }

    long nInputRows = input.size(dimh);
    long nInputCols = input.size(dimw);

    /////////check depth map shape /////////

    int ndim_depth = input_depth.ndimension();
    int dimf_depth = 0;
    int dimh_depth = 1;
    int dimw_depth = 2;

    if (ndim_depth == 4) {
        dimf_depth++;
        dimh_depth++;
        dimw_depth++;
    }

    if(ndim_depth != 3 && ndim_depth != 4) {
         throw std::invalid_argument("depthavgpool: 3D input depth tensor expected but got: " + std::to_string(ndim));
    }

    long inputHeight_depth = input_depth.size(dimh_depth);
    long inputWidth_depth = input_depth.size(dimw_depth);

    if (input_depth.size(1) != 1){
         throw std::invalid_argument("depthavgpool: input depth should have only 1 channel");
    }

    if (!(nInputRows == inputHeight_depth && nInputCols == inputWidth_depth)){
        throw std::invalid_argument(
            string_format("depthavgpool: input image and input depth should be the same size, but got: image(%d,%d), depth(%d,%d)",
                nInputRows, nInputCols, inputHeight_depth, inputWidth_depth));
    }
}

void shape_check(torch::Tensor input, torch::Tensor input_depth,
    torch::Tensor depthweightcount, torch::Tensor gradOutput,
    int kH, int kW, int dH, int dW, int padH, int padW) {

    shape_check_forward(input, input_depth, kH, kW, dH, dW, padH, padW);

    if(depthweightcount.size(1) != 1){
        throw std::invalid_argument("depthavgpool: input depth should have only 1 channel");
    }

    int ndim = input.ndimension();
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
    }

    long nInputRows = input.size(dimh);
    long nInputCols = input.size(dimw);

    /////////check depth map shape /////////

    int ndim_depth = input_depth.ndimension();
    int dimf_depth = 0;
    int dimh_depth = 1;
    int dimw_depth = 2;

    if (ndim_depth == 4) {
        dimf_depth++;
        dimh_depth++;
        dimw_depth++;
    }

    long inputHeight_depth = input_depth.size(dimh_depth);
    long inputWidth_depth = input_depth.size(dimw_depth);

    if(!(inputHeight_depth == depthweightcount.size(2) && inputWidth_depth == depthweightcount.size(3))){
        throw std::invalid_argument(
            string_format("depthavgpool: input depth and input depthweightcount should be the same size, but got: weightcount(%d,%d), depth(%d,%d)",
                depthweightcount.size(dimh_depth), depthweightcount.size(dimw_depth), inputHeight_depth, inputWidth_depth));
    }

//////////////////////////////////////////

	long nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
	long nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
	long nInputPlane = input.size(dimh-1);
	long nOutputPlane = nInputPlane;

    if (padW || padH){
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
        if ((nOutputRows - 1)*dH >= nInputRows + padH)
            --nOutputRows;
        if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
            --nOutputCols;
    }

    if (nOutputCols < 1 || nOutputRows < 1)
        throw std::invalid_argument(
            string_format("depthavgpool: Given input size: (%dx%dx%d). "
                "Calculated output size: (%dx%dx%d). Output size is too small",
                nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols));

    if(gradOutput.size(dimf) != nOutputPlane) {
        throw std::invalid_argument(
            string_format("depthavgpool: invalid number of gradOutput planes, expected: %d, but got: %d",
                nOutputPlane, gradOutput.size(dimf)));
    }

    if(!(gradOutput.size(dimh) == nOutputRows && gradOutput.size(dimw) == nOutputCols)){
        throw std::invalid_argument(
            string_format("depthavgpool: invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d",
                nOutputRows, nOutputCols, gradOutput.size(dimh), gradOutput.size(dimw)));
    }
}


torch::Tensor depthavgpooling_forward_cuda(
    torch::Tensor input,
    torch::Tensor input_depth,
    torch::Tensor depthweightcount,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool useDepth) {

    CHECK_INPUT(input);
    CHECK_INPUT(input_depth);
    CHECK_INPUT(depthweightcount);

    shape_check_forward(input, input_depth, kH, kW, dH, dW, padH, padW);

    int batch = 1;
    long nInputCols, nInputRows, nInputPlane, batchSize;
    long nOutputCols, nOutputRows;

    if (input.ndimension() == 3) {
        nInputCols = input.size(2);
        nInputRows = input.size(1);
        nInputPlane = input.size(0);
        batchSize = 1;
        batch = 0;
        input = input.reshape({1, input.size(0), input.size(1), input.size(2)});
        input_depth = input_depth.reshape({1, input_depth.size(0), input_depth.size(1), input_depth.size(2)});
    }else{
        nInputCols = input.size(3);
        nInputRows = input.size(2);
        nInputPlane = input.size(1);
        batchSize = input.size(0);
    }

    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;

    //  long batchSize = input->size[0];

    if (padW || padH){
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
        if ((nOutputRows - 1)*dH >= nInputRows + padH)
            --nOutputRows;
        if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
            --nOutputCols;
    }

    //  input = THCudaTensor_newContiguous(state, input);
    //  float* input_data = THCudaTensor_data(state, input);
    //  float* input_depth_data = THCudaTensor_data(state, input_depth);

    torch::Tensor output = torch::zeros({batchSize, nInputPlane, nOutputRows, nOutputCols}, torch::kCUDA);
    depthweightcount = depthweightcount.reshape({batchSize, 1, nInputRows, nInputCols});

    //  float* output_data = THCudaTensor_data(state, output);
    //  float* depthweightcount_data = THCudaTensor_data(state, depthweightcount);

    torch::Tensor input_n;
    torch::Tensor depth_n;
    torch::Tensor depthweightcount_n;
    torch::Tensor output_n;

    for (int elt = 0; elt < batchSize; elt++) {
        input_n = input.select(0, elt);
        depth_n = input_depth.select(0, elt);
        depthweightcount_n = depthweightcount.select(0, elt);
        output_n = output.select(0, elt);

        int count = output_n.numel();

        AvePoolForward(count, input_n, depth_n,
            nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
            kH, kW, dH, dW, padH, padW, output_n, depthweightcount_n);
    }

    if(batch == 0){
        output = output.reshape({nInputPlane, nOutputRows, nOutputCols});
        input = input.reshape({nInputPlane, nInputRows, nInputCols});
    }

    return output;
}


torch::Tensor depthavgpooling_backward_cuda(
    torch::Tensor input,
    torch::Tensor input_depth,
    torch::Tensor depthweightcount,
    torch::Tensor gradOutput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool useDepth) {

    CHECK_INPUT(input);
    CHECK_INPUT(input_depth);
    CHECK_INPUT(depthweightcount);
    CHECK_INPUT(gradOutput);

    shape_check(input, input_depth, depthweightcount, gradOutput, kH, kW, dH, dW,
        padH, padW);

    long nInputCols, nInputRows, nInputPlane, batchSize;
    long nOutputCols, nOutputRows;
    int dimCol = 2;
    int dimRow = 1;

    int batch = 1;
    if (input.ndimension() == 3) {
        nInputPlane = input.size(0);
        batchSize = 1;
        batch = 0;
        input = input.reshape({1, input.size(0), input.size(1),input.size(2)});
        gradOutput = gradOutput.reshape({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
    }else{
        dimCol = 3;
        dimRow = 2;
        nInputPlane = input.size(1);
        batchSize = input.size(0);
    }
    nInputCols = input.size(dimCol);
    nInputRows = input.size(dimRow);

    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
    if (padW || padH){
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
        if ((nOutputRows - 1)*dH >= nInputRows + padH)
            --nOutputRows;
        if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
            --nOutputCols;
    }

    //  THCUNN_check_dim_size(state, gradOutput, input->nDimension, dimRow, nOutputRows);
    //  THCUNN_check_dim_size(state, gradOutput, input->nDimension, dimCol, nOutputCols);

    if(input_depth.size(0) != batchSize){
        throw std::invalid_argument("depthavgpool: invalid batch size of input depth");
    }

    torch::Tensor gradInput = torch::zeros_like(input);

    //  float* input_depth_data = THCudaTensor_data(state, input_depth);
    //  float* depthweightcount_data = THCudaTensor_data(state, depthweightcount);

    torch::Tensor gradInput_n;
    torch::Tensor depth_n;
    torch::Tensor gradOutput_n;
    torch::Tensor depthweightcount_n;

    for (int elt = 0; elt < batchSize; elt++) {
        gradInput_n = gradInput.select(0, elt);
        depth_n = input_depth.select(0, elt);
        gradOutput_n = gradOutput.select(0, elt);
        depthweightcount_n = depthweightcount.select(0, elt);

        int count = gradInput_n.numel();

        AvePoolBackward(count, gradOutput_n, depth_n, depthweightcount_n,
            nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
            kH, kW, dH, dW, padH, padW,
            gradInput_n);

    }

    if (batch == 0) {
        gradOutput = gradOutput.reshape({nInputPlane, nOutputRows, nOutputCols});
        input = input.reshape({nInputPlane, nInputRows, nInputCols});
        input_depth = input_depth.reshape({1, nInputRows, nInputCols});
        gradInput = gradInput.reshape({nInputPlane, nInputRows,nInputCols});
    }

    return gradInput;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &depthavgpooling_forward_cuda, "DepthAvgPooling forward (CUDA)");
  m.def("backward", &depthavgpooling_backward_cuda, "DepthAvgPooling backward (CUDA)");
}