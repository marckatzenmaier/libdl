//
// Created by marc on 03.06.19.
//

#include "libdl/math_functions.h"
#include <iostream>
#include "array"
using namespace std;
using namespace Eigen;


Eigen::Tensor<float, 4, RowMajor> conv2d_NHWC(const Eigen::Tensor<float, 4, RowMajor> &input, const Eigen::Tensor<float, 4, RowMajor> &kernel){
    /**
     * Calculates the conv2d(tensorflow calls it like that) for convolutional layers
     * input is a tensor<float, 4> with Batch,Height,Width,Channel
     * kernel is a tensor<float, 4> with output_channels, width, height, channels_input
     *
     * */
    int N = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);
    //int C=input.dimension(3);
    int kernel_h = kernel.dimension(0);
    int kernel_w = kernel.dimension(1);
    int kernel_ci=kernel.dimension(2);
    int kernel_co=kernel.dimension(3);
    int out_h;
    int out_w;
    out_h = H - (kernel_h-1);
    out_w = W - (kernel_w-1);

    Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

    typedef typename Eigen::internal::traits<
            Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index TensorIndex;

    Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
    contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

    Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
    pre_contract_dims[0] = N*out_h*out_w;
    pre_contract_dims[1] = kernel_h * kernel_w * kernel_ci;

    Eigen::DSizes<TensorIndex, 2> kernel_dims;
    kernel_dims[0] = pre_contract_dims[1];
    kernel_dims[1] = kernel_co;

    output = input
            .extract_image_patches(
                    kernel_w,
                    kernel_h,
                    1,//stride,
                    1,//stride,
                    1,//dilation,
                    1,//dilation,
                    1,
                    1,
                    0,//pad_l(),
                    0,//pad_r(),
                    0,//pad_t(),
                    0,//pad_b(),
                    0)
            .reshape(pre_contract_dims)
            .contract(kernel.reshape(kernel_dims), contract_dims)
            .reshape(output.dimensions());
    return output;
}

Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC_backprop_kernel(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input,
                                                                     const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad){

    const Eigen::Tensor<float, 4, Eigen::RowMajor> &kernel = grad;
    int N = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);
    int C = input.dimension(3);
    //int grad_n=kernel.dimension(0);
    int grad_h = kernel.dimension(1);
    int grad_w = kernel.dimension(2);
    int grad_c=kernel.dimension(3);
    int filter_h;
    int filter_w;
    filter_h = H - (grad_h-1);
    filter_w = W - (grad_w-1);

    Eigen::Tensor<float, 4, RowMajor> output(filter_h, filter_w, C, grad_c);

    typedef typename Eigen::internal::traits<
            Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index TensorIndex;

    Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
    contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

    Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
    pre_contract_dims[1] = N*grad_h*grad_w;
    pre_contract_dims[0] = C*filter_h*filter_w;

    Eigen::DSizes<TensorIndex, 2> kernel_dims;
    kernel_dims[0] = pre_contract_dims[1];
    kernel_dims[1] = grad_c;

    std::array<int,5> swap = {1,4,0,2,3};
    output = input
            .extract_image_patches(
                    grad_w,
                    grad_h,
                    1,//stride,
                    1,//stride,
                    1,//dilation,
                    1,//dilation,
                    1,
                    1,
                    0,//pad_l(),
                    0,//pad_r(),
                    0,//pad_t(),
                    0,//pad_b(),
                    0).eval().shuffle(swap).reshape(pre_contract_dims)
                    .contract(kernel.reshape(kernel_dims), contract_dims).reshape(output.dimensions());

    return output;

}


Eigen::Tensor<float, 4, Eigen::RowMajor> conv2d_NHWC_backprop_input(const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad,
                                                                    const Eigen::Tensor<float, 4, Eigen::RowMajor> &kernel){
    int N = grad.dimension(0);
    int C = kernel.dimension(2);
    int kernel_h = kernel.dimension(0);
    int kernel_w = kernel.dimension(1);
    int kernel_ci=kernel.dimension(2);
    int kernel_co=kernel.dimension(3);
    int out_h= grad.dimension(1);
    int out_w= grad.dimension(2);
    int H = kernel_h + out_h -1;
    int W = kernel_w + out_w -1;

    Eigen::Tensor<float, 4, RowMajor> output(N, H, W, C);

    typedef typename Eigen::internal::traits<
            Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index TensorIndex;

    Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
    contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

    Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
    pre_contract_dims[0] = N*H*W;
    pre_contract_dims[1] = kernel_h * kernel_w * kernel_co ;// * kernel_ci;

    Eigen::DSizes<TensorIndex, 2> kernel_dims;
    kernel_dims[0] = pre_contract_dims[1];
    kernel_dims[1] = /*kernel_co* */ kernel_ci;
    Eigen::array<pair<int, int>, 4> paddings;
    paddings[0] = make_pair(0,0);
    paddings[1] = make_pair(kernel_h-1, kernel_h-1);//left, right
    paddings[2] = make_pair(kernel_w-1, kernel_w-1);//top, bottom
    paddings[3] = make_pair(0,0);
    Eigen::array<bool, 4> reverse({true, true, false, false});

    std::array<int,5> swap = {0,1,4,2,3};
    std::array<int,4> swap_kernel = {3,0,1,2};

    output = grad.pad(paddings)
            .extract_image_patches(
                    kernel_w,
                    kernel_h,
                    1,//stride,
                    1,//stride,
                    1,//dilation,
                    1,//dilation,
                    1,
                    1,
                    0,//pad_l(),
                    0,//pad_r(),
                    0,//pad_t(),
                    0,//pad_b(),
                    0).eval().shuffle(swap)
            .reshape(pre_contract_dims)
            .contract(kernel.reverse(reverse).eval().shuffle(swap_kernel).eval().reshape(kernel_dims), contract_dims).reshape(output.dimensions());
    return output;
}