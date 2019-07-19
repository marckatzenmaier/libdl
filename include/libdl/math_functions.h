//
// Created by marc on 03.06.19.
//

#ifndef TEST_MATH_FUNCTIONS_H
#define TEST_MATH_FUNCTIONS_H
#include "unsupported/Eigen/CXX11/Tensor"

/**
 * performs the forward pass of the 2d convolution
 * @param input input over which is convoluted
 * @param kernel kernel with which is convoluted
 * @return the result of the convolution
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input,
        const Eigen::Tensor<float, 4, Eigen::RowMajor> &kernel);

/**
 * calculates the gradient of the 2d convolution with respect to the input
 * @param grad gradient which should be backpropagated
 * @param kernel kernel which was used for the forward pass
 * @return gradient with respect to the input
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC_backprop_input(const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad,
        const Eigen::Tensor<float, 4, Eigen::RowMajor> &kernel);

/**
 * calculates the gradient of the 2d convolution with respect to the kernel
 * @param input the input on which was convoluted
 * @param grad the gradient which should be backpropagated
 * @return the gradient with respect to the kernel
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC_backprop_kernel(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input,
        const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad);

/**
 * calculates an elementwise sigmoid
 * @param input the tensor which entries are the input to the sigmoid
 * @return the result tensor of the sigmoid calculation
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> sigmoid(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input);

/**
 * the forward pass of the average pooling
 * @param input the tensor on which the average pooling should be performed
 * @param kernel_h height of the pooling kernel
 * @param kernel_w width of the pooling kernel
 * @param stride_h stride in height direction
 * @param stride_w stride in width direction
 * @return result of the average pooling
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> pool_average(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, int kernel_h, int kernel_w, int stride_h, int stride_w);

/**
 * calculates the gradient of the average pooling
 * @param grad gradient which should be backporpagated
 * @param kernel_h height of the pooling kernel which was used for the forward pass
 * @param kernel_w width of the pooling kernel which was used for the forward pass
 * @param stride_h stride in height direction which was used for the forward pass
 * @param stride_w stride in width direction which was used for the forward pass
 * @return the gradient with respekt to the input of the pooling
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> pool_average_backward(const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad, int kernel_h, int kernel_w, int stride_h, int stride_w);

/**
 * perorms nearest neighbour upsampling
 * @param input the data which should be upsampled
 * @param height factor with which the input height is multiplied
 * @param width factor with which the input width is multiplied
 * @return the upsampled data
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> nn_upscale(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, int height, int width);
/**
 * calculates the softmax of the input
 * @param input
 * @return the result of the softmax function
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> softmax(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input);
/**
 * calculates the gradient of the softmax with respect to the inputs
 * @param input data used for the forward pass of the softmax
 * @param grad gradient which should be backpropagated
 * @return gradient of with respect to the inputs
 */
Eigen::Tensor<float, 4, Eigen::RowMajor> softmax_backward(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad);

/**
 * takes a 2d tensor in which the first dimension represents the batch size and diagonalizes the second dim to a 2d tensor
 * @param input tensor(batch, dim_of_vector)
 * @return tensor(batch, dim_of_vector, dim_of_vector)
 */
Eigen::Tensor<float, 3, Eigen::RowMajor> diagonalize(Eigen::Tensor<float, 2, Eigen::RowMajor> &input);


#endif //TEST_MATH_FUNCTIONS_H
