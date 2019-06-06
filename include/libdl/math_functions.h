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
#endif //TEST_MATH_FUNCTIONS_H
