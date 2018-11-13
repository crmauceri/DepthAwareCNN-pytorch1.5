#include <THC/THC.h>

#include "depthconv_cuda_kernel.h"

extern THCState *state;

void shape_check(THCState *state, THCudaTensor *input, THCudaTensor *input_depth,
                 THCudaTensor *gradOutput, THCudaTensor *weight, THCudaTensor *bias, int kH, int kW,
                 int dH, int dW, int padH, int padW, int dilationH,
                 int dilationW) {

  THArgCheck(THCudaTensor_nDimension(state, weight) == 4, 5,
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             THCudaTensor_nDimension(state, weight));

  THArgCheck(THCudaTensor_isContiguous(state, weight), 5,
             "weight tensor has to be contiguous");

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
             kH, kW);

  THArgCheck((THCudaTensor_size(state, weight, 2) == kH && THCudaTensor_size(state, weight, 3) == kW), 9,
             "kernel size should be consistent with weight, but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, THCudaTensor_size(state, weight, 2), THCudaTensor_size(state, weight, 3));

  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH,
             dW);

  THArgCheck(
      dilationW > 0 && dilationH > 0, 14,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  //////////// check bias //////////////////

  THArgCheck(!bias || THCudaTensor_isContiguous(state, bias), 5,
             "bias tensor has to be contiguous");

  if (bias != NULL) {
//    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[0]);
    THArgCheck(THCudaTensor_nDimension(state, bias) == 1, 6,
             "Need bias of dimension %d but got %d", 1, THCudaTensor_nDimension(state, bias));
    THArgCheck(THCudaTensor_size(state, bias, 0) == THCudaTensor_size(state, weight, 0), 6,
             "Need bias of size %d but got %d", THCudaTensor_size(state, weight, 0), THCudaTensor_size(state, bias, 0));
  }
//////////////////////////////////////////

  int ndim = THCudaTensor_nDimension(state, input);
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THArgCheck(ndim == 3 || ndim == 4, 2,
             "3D or 4D input tensor expected but got: %s", ndim);

  long nInputPlane = THCudaTensor_size(state, weight, 1);
  long inputHeight = THCudaTensor_size(state, input, dimh);
  long inputWidth = THCudaTensor_size(state, input, dimw);
  long nOutputPlane = THCudaTensor_size(state, weight, 0);

  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  THArgCheck((inputHeight >= kH && inputWidth >= kW), 2,
             "input image is smaller than kernel");

/////////check depth map shape /////////

  int ndim_depth = THCudaTensor_nDimension(state, input);
  int dimf_depth = 0;
  int dimh_depth = 1;
  int dimw_depth = 2;

  if (ndim_depth == 4) {
    dimf_depth++;
    dimh_depth++;
    dimw_depth++;
  }

  THArgCheck(ndim_depth == 3 || ndim_depth == 4, 3,
             "3D input depth tensor expected but got: %s", ndim);

  //long inputHeight_depth = input_depth->size[dimh_depth];
  //long inputWidth_depth = input_depth->size[dimw_depth];
  long inputHeight_depth = THCudaTensor_size(state, input_depth, dimh_depth);
  long inputWidth_depth = THCudaTensor_size(state, input_depth, dimw_depth);

  THArgCheck(THCudaTensor_size(state, input_depth, 1) == 1, 3,
             "input depth should have only 1 channel",
             nInputPlane, THCudaTensor_size(state, input, 1));

  THArgCheck((inputHeight == inputHeight_depth && inputWidth == inputWidth_depth), 3,
             "input image and input depth should be the same size");
//////////////////////////////////////////

  if (gradOutput != NULL) {
    THArgCheck(THCudaTensor_size(state, gradOutput, dimf) == nOutputPlane, 4,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, THCudaTensor_size(state, gradOutput, dimf));
    THArgCheck((THCudaTensor_size(state, gradOutput, dimh) == outputHeight &&
                THCudaTensor_size(state, gradOutput, dimw) == outputWidth),
               4, "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", outputHeight, outputWidth,
               THCudaTensor_size(state, gradOutput, dimh), THCudaTensor_size(state, gradOutput, dimw));
  }
}

int depthconv_forward_cuda(THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *weight, THCudaTensor *bias, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, input_depth, weight, output, columns, ones, bias));

  shape_check(state, input, input_depth, NULL, weight, bias, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (THCudaTensor_nDimension(state, input) == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, THCudaTensor_size(state, input, 0), THCudaTensor_size(state, input, 1),
                          THCudaTensor_size(state, input, 2));
    THCudaTensor_resize4d(state, input_depth, 1, THCudaTensor_size(state, input_depth, 0), THCudaTensor_size(state, input_depth, 1),
                          THCudaTensor_size(state, input_depth, 2));
  }

  long batchSize = THCudaTensor_size(state, input, 0);
  long nInputPlane = THCudaTensor_size(state, input, 1);
  long inputHeight = THCudaTensor_size(state, input, 2);
  long inputWidth = THCudaTensor_size(state, input, 3);

  long nOutputPlane = THCudaTensor_size(state, weight, 0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  bias = bias ? THCudaTensor_newContiguous(state, bias) : bias;
  THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight,
                        outputWidth);

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

  if (THCudaTensor_nDimension(state, ones) != 2 ||
      THCudaTensor_size(state, ones, 0) * THCudaTensor_size(state, ones, 1) < outputHeight * outputWidth) {
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *depth_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {

    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);


    // Do bias first
     long m_ = nOutputPlane;
     long n_ = outputHeight * outputWidth;
     long k_ = 1;

     if (bias) {
       THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
                        THCudaTensor_data(state, ones), k_,
                        THCudaTensor_data(state, bias), k_, 0.0f,
                        THCudaTensor_data(state, output_n), n_);
     } else {
       THCudaTensor_zero(state, output_n);
     }

    depthconv_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n), THCudaTensor_data(state, depth_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = THCudaTensor_size(state, columns, 1);
    long k = nInputPlane * kH * kW;

    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     THCudaTensor_data(state, columns), n,
                     THCudaTensor_data(state, weight), k, 1.0f,
                     THCudaTensor_data(state, output_n), n);
  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, depth_n);
  THCudaTensor_free(state, output_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
  THCudaTensor_free(state, weight);

  if (bias) THCudaTensor_free(state, bias);

  return 1;
}

int depthconv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, input_depth, gradOutput, weight, columns, gradInput));

  shape_check(state, input, input_depth, gradOutput, weight, NULL, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (THCudaTensor_nDimension(state, input) == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, THCudaTensor_size(state, input, 0), THCudaTensor_size(state, input, 1),
                          THCudaTensor_size(state, input, 2));
    THCudaTensor_resize4d(state, gradOutput, 1, THCudaTensor_size(state, gradOutput, 0),
                          THCudaTensor_size(state, gradOutput, 1), THCudaTensor_size(state, gradOutput, 2));
  }

  long batchSize = THCudaTensor_size(state, input, 0);
  long nInputPlane = THCudaTensor_size(state, input, 1);
  long inputHeight = THCudaTensor_size(state, input, 2);
  long inputWidth = THCudaTensor_size(state, input, 3);

  long nOutputPlane = THCudaTensor_size(state, weight, 0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((THCudaTensor_size(state, input_depth, 0) == batchSize), 3, "invalid batch size of input depth");

  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight,
                        inputWidth);

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

//  printf("columns size: %d,%d\n", columns->size[0],columns->size[1]);

  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *input_depth_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, input_depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    long m = nInputPlane * kW * kH;
    long n = THCudaTensor_size(state, columns, 1);
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     THCudaTensor_data(state, gradOutput_n), n,
                     THCudaTensor_data(state, weight), m, 0.0f,
                     THCudaTensor_data(state, columns), n);

    depthconv_col2im(
        THCState_getCurrentStream(state), THCudaTensor_data(state, columns),
        THCudaTensor_data(state, input_depth_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, gradInput_n));
  }

  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, input_depth_n);
  THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, input_depth, 1, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight,
                          inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, weight);

  return 1;
}

int depthconv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, THCudaTensor *gradBias,
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, input_depth, gradOutput,
                                         gradWeight, gradBias, columns, ones));

  shape_check(state, input, input_depth, gradOutput, gradWeight, gradBias, kH, kW, dH, dW,
              padH, padW, dilationH, dilationW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  int batch = 1;
  if (THCudaTensor_nDimension(state, input) == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, THCudaTensor_size(state, input, 0), THCudaTensor_size(state, input, 1),
                          THCudaTensor_size(state, input, 2));
    THCudaTensor_resize4d(state, gradOutput, 1, THCudaTensor_size(state, gradOutput, 0),
                          THCudaTensor_size(state, gradOutput, 1), THCudaTensor_size(state, gradOutput, 2));
  }

  long batchSize = THCudaTensor_size(state, input, 0);
  long nInputPlane = THCudaTensor_size(state, input, 1);
  long inputHeight = THCudaTensor_size(state, input, 2);
  long inputWidth = THCudaTensor_size(state, input, 3);

  long nOutputPlane = THCudaTensor_size(state, gradWeight, 0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;


  // Define a buffer of ones, for bias accumulation
  if (THCudaTensor_nDimension(state, ones) != 2 ||
      THCudaTensor_size(state, ones, 0) * THCudaTensor_size(state, ones, 1) < outputHeight * outputWidth) {
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *depth_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    depthconv_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n),
        THCudaTensor_data(state, depth_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = THCudaTensor_size(state, columns, 1);

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     THCudaTensor_data(state, columns), k,
                     THCudaTensor_data(state, gradOutput_n), k, 1.0f,
                     THCudaTensor_data(state, gradWeight), n);


    // Do Bias:
    // M,N,K are dims of matrix A and B
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias)
        THCudaBlas_Sgemv(
          state,
          't',
          k_, m_,
          scale,
          THCudaTensor_data(state, gradOutput_n), k_,
          THCudaTensor_data(state, ones), 1, 1.0f,
          THCudaTensor_data(state, gradBias), 1);


  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, depth_n);
  THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
  THCudaTensor_free(state, gradOutput);
  return 1;
}
