//
// Created by marc on 03.06.19.
//

#ifndef TEST_MATH_FUNCTIONS_H
#define TEST_MATH_FUNCTIONS_H
#include "unsupported/Eigen/CXX11/Tensor"


Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input,
        const Eigen::Tensor<float, 4, Eigen::RowMajor> &kernel);
Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC_backprop_input(const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad,
        const Eigen::Tensor<float, 4, Eigen::RowMajor> &kernel);
Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC_backprop_kernel(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input,
        const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad);
Eigen::Tensor<float, 4, Eigen::RowMajor> sigmoid(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input);
Eigen::Tensor<float, 4, Eigen::RowMajor> pool_average(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, int kernel_h, int kernel_w, int stride_h, int stride_w);
Eigen::Tensor<float, 4, Eigen::RowMajor> pool_average_backward(const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad, int kernel_h, int kernel_w, int stride_h, int stride_w);
Eigen::Tensor<float, 4, Eigen::RowMajor> nn_upscale(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, int height, int width);
Eigen::Tensor<float, 4, Eigen::RowMajor> softmax(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input);
Eigen::Tensor<float, 4, Eigen::RowMajor> softmax_backward(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad);

Eigen::Tensor<float, 3, Eigen::RowMajor> diagonalize(Eigen::Tensor<float, 2, Eigen::RowMajor> &input);


#endif //TEST_MATH_FUNCTIONS_H
