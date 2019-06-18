//
// Created by marc on 03.06.19.
//

#include "libdl/math_functions.h"
#include "libdl/graph_node.h"
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

Eigen::Tensor<float, 4, Eigen::RowMajor> sigmoid(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input){
    return 1/(1+(-input).exp());
}


Eigen::Tensor<float, 4, Eigen::RowMajor> pool_average(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, int kernel_h, int kernel_w, int stride_h, int stride_w){

    int N = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);
    int C=input.dimension(3);
    int out_h;
    int out_w;
    out_h = (H - (kernel_h-stride_h))/stride_h;
    out_w = (W - (kernel_w-stride_w))/stride_w;
    if (out_h != (H - (kernel_h-stride_h))/(float)stride_h || out_w != (W - (kernel_w-stride_w))/(float)stride_w){
        cout<<"error"<<endl; //todo throw error
    }

    Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, C);//b,h,w,c

    typedef typename Eigen::internal::traits<
            Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index TensorIndex;


    Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
    pre_contract_dims[0] = N*out_h*out_w*C;
    pre_contract_dims[1] = kernel_h * kernel_w;

    std::array<int,5> swap = {0,1,4,2,3};//maybe useless since mean_dim applyable
    std::array<int,1> mean_dim = {1};
    Eigen::Tensor<float, 2, RowMajor> output2;

    output = input
            .extract_image_patches(
                    kernel_w,
                    kernel_h,
                    stride_w,
                    stride_h,
                    1,//dilation,
                    1,//dilation,
                    1,
                    1,
                    0,//pad_l(),
                    0,//pad_r(),
                    0,//pad_t(),
                    0,//pad_b(),
                    0).eval().shuffle(swap)
            .reshape(pre_contract_dims).mean(mean_dim)
            .reshape(output.dimensions());
    return output;
}

Eigen::Tensor<float, 4, Eigen::RowMajor> nn_upscale(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input, int height, int width){
    int N = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);
    int C=input.dimension(3);
    Eigen::Tensor<float, 4, Eigen::RowMajor> output(N,height*H,width*W,C);
    Eigen::Tensor<float, 4, Eigen::RowMajor> output_h(N,H,W*width,C);
    for(int i = 0;i<N*H*W;i++){
        for(int j = 0;j<width;j++){
            memcpy(output_h.data() + i * width * C + j * C, input.data()+i*C, sizeof(float) * C);
        }
    }
    for(int b = 0; b<N; b++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < height; j++) {
                memcpy(output.data() + j *  width * W * C + i * width * W * height * C+ b * H * W * C * width * height,
                       output_h.data() + i * width * W * C+ b * H * W * C * width,
                       sizeof(float) * width * W * C);
            }
        }
    }
    return output;
}

Eigen::Tensor<float, 4, Eigen::RowMajor> pool_average_backward(const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad, int kernel_h, int kernel_w, int stride_h, int stride_w){
    //todo only hacked so it works backwards for int kernel_h, int kernel_w, int stride_h, int stride_w == 2 but more is not needed for mnist
    std::array<int,4> broadcast({(int)grad.dimension(0),(int)grad.dimension(1)*stride_h,(int)grad.dimension(2)*stride_w,(int)grad.dimension(3)});
    Eigen::Tensor<float, 4, Eigen::RowMajor> factor(1,1,1,1);
    factor.setValues({{{{0.25}}}});
    Eigen::Tensor<float, 4, Eigen::RowMajor> output = nn_upscale(grad,stride_h,stride_w)*factor.broadcast(broadcast);
    return output;

}


Eigen::Tensor<float, 4, Eigen::RowMajor> softmax(const Eigen::Tensor<float, 4, Eigen::RowMajor> &input){
    //todo make stable
    std::array<int,1> sum = {3};
    std::array<int,4> reshape = {(int)input.dimension(0),(int)input.dimension(1),(int)input.dimension(2),1};
    std::array<int, 4> bcast = {1,1,1,(int)input.dimension(3)};
    Eigen::Tensor<float, 4, Eigen::RowMajor> exp_tensor = (input-input.maximum(sum).reshape(reshape).broadcast(bcast)).exp();
    Eigen::Tensor<float, 4, Eigen::RowMajor> output = exp_tensor / exp_tensor.sum(sum).reshape(reshape).broadcast(bcast);
    return output;
}

Eigen::Tensor<float, 3, Eigen::RowMajor> diagonalize(Eigen::Tensor<float, 2, Eigen::RowMajor> &input){
    int N = input.dimension(0);
    int size = input.dimension(1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(N, size, size);
    output.setZero();
    for(int b=0;b<N;b++) {
        for (int i = 0; i < size; i++) {
            output.data()[b*size*size+i * (size + 1)] = input.data()[b*size+i];
        }
    }
    return output;
}

Eigen::Tensor<float, 4, Eigen::RowMajor> softmax_backward(Eigen::Tensor<float, 4, Eigen::RowMajor> &input, const Eigen::Tensor<float, 4, Eigen::RowMajor> &grad){
    int C = input.dimension(3);
    int NHW = input.dimension(0)*input.dimension(1)*input.dimension(2);

    std::array<int,3> input_dot_1 = {NHW, C, 1};
    std::array<int,3> input_dot_b1 = {1, 1, C};
    std::array<int,3> input_dot_2 = {NHW, 1, C};
    std::array<int,3> input_dot_b2 = {1, C, 1};
    Eigen::Tensor<float, 3, Eigen::RowMajor> dot = input.reshape(input_dot_1).broadcast(input_dot_b1).eval() *
            input.reshape(input_dot_2).broadcast(input_dot_b2).eval();

    std::array<int,2> reshape_input = {NHW,C};
    Eigen::Tensor<float, 2, Eigen::RowMajor> reshaped_input = input.reshape(reshape_input);

    Eigen::Tensor<float, 3, Eigen::RowMajor> jacobi = diagonalize(reshaped_input) - dot;

    std::array<int,3> reshape_grad = {NHW,C,1};
    std::array<int,3> grad_b = {1, 1,C};
    std::array<int,1> grad_sum = {1};
    Eigen::Tensor<float, 4, Eigen::RowMajor> output = (jacobi * grad.reshape(reshape_grad).broadcast(grad_b)).sum(grad_sum).reshape(input.dimensions());

    return output ;


}