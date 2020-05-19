template <typename scalar_t>
void depthconv_im2col(const scalar_t *data_im,
                       const scalar_t *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, scalar_t *data_col);

template <typename DTscalar_type>
void depthconv_col2im(const scalar_t *data_col,
                       const scalar_t *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, scalar_t *grad_im);

