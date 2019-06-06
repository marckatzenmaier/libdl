//
// Created by marc on 03.06.19.
//

#include "catch2/catch.hpp"
#include "libdl/math_functions.h"
#include <iostream>
using namespace Eigen;

SCENARIO( "Test conv2d_NHWC", "[Math]"){
    WHEN("input simple kernel 2x2x1x1"){
        int N = 1, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1},{2},{3}},
                                 {{4},{5},{6}},
                                 {{7},{8},{9}}}});

        kernel.setValues({{{{1}},{{1}}},
                          {{{1}},{{1}}}});

        output = conv2d_NHWC(input, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{12},{16}},
                                           {{24},{28}}}});

        THEN("output should be like expected"){
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values-output).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("input simple kernel 2x2x1x1 but asymetric"){
        int N = 1, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1},{2},{3}},
                          {{4},{5},{6}},
                          {{7},{8},{9}}}});

        kernel.setValues({{{{1}},{{1}}},
                          {{{2}},{{1}}}});

        output = conv2d_NHWC(input, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{16},{21}},
                                    {{31},{36}}}});
        THEN("output should be like expected"){
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values-output).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("input simple kernel 2x2x1x1 with batch == 2 "){
        int N = 2, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1},{2},{3}},
                          {{4},{5},{6}},
                          {{7},{8},{9}}},
                         {{{2},{2},{2}},
                          {{4},{4},{4}},
                          {{6},{6},{6}}}
                        });

        kernel.setValues({{{{1},{1}},
                           {{1},{1}}}});

        output = conv2d_NHWC(input, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{12},{16}},
                                    {{24},{28}}},
                                   {{{12},{12}},
                                    {{20},{20}}}});
        THEN("output should be like expected"){
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values-output).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("input kernel 2x2x2x1 input channel == 2"){
        int N = 1, H = 3, W = 3, C=2, kernel_h = 2, kernel_w = 2, kernel_ci=2,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1,1},{2,2},{3,3}},
                          {{4,4},{5,5},{6,6}},
                          {{7,7},{8,8},{9,9}}}});

        kernel.setValues({{{{1},{1}},{{1},{1}}},
                          {{{1},{1}},{{1},{1}}}});

        output = conv2d_NHWC(input, kernel);


        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{24},{32}},
                                    {{48},{56}}}});
        THEN("output should be like expected"){
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values-output).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("input kernel 2x2x1x3 output channel == 3"){
        int N = 1, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=3, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1},{2},{3}},
                          {{4},{5},{6}},
                          {{7},{8},{9}}}});

        kernel.setValues({{{{1,2,3}},{{1,2,3}}},
                          {{{1,2,3}},{{1,2,3}}}});

        output = conv2d_NHWC(input, kernel);


        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{12,24,36},{16,32,48}},
                                    {{24,48,72},{28,56,84}}}});
        THEN("output should be like expected"){
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values-output).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("input simple kernel 2x2x2x3 and batch == 2"){
        int N = 2, H = 3, W = 3, C=2, kernel_h = 2, kernel_w = 2, kernel_ci=2,kernel_co=3, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1,1},{2,2},{3,3}},
                          {{4,4},{5,5},{6,6}},
                          {{7,7},{8,8},{9,9}}},
                         {{{2,2},{2,2},{2,2}},
                          {{4,4},{4,4},{4,4}},
                          {{6,6},{6,6},{6,6}}}
                        });

        kernel.setValues({{{{1,2,3},{1,2,3}},
                           {{1,2,3},{1,2,3}}},
                          {{{1,2,3},{1,2,3}},
                           {{1,2,3},{1,2,3}}}});
        output = conv2d_NHWC(input, kernel);


        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{24,48,72},{32,64,96}},
                                    {{48,96,144},{56,112,168}}},
                                   {{{24,48,72},{24,48,72}},
                                    {{40,80,120},{40,80,120}}}});
        THEN("output should be like expected"){
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values-output).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("input simple kernel 2x3x1x1 asymetric kernel dimensions"){
        int N = 1, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 3, kernel_ci=1,kernel_co=1, out_h=2, out_w=1;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> output(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1},{2},{3}},
                          {{4},{5},{6}},
                          {{7},{8},{9}}}});

        kernel.setValues({{{{1}},{{2}},{{1}}},
                          {{{1}},{{2}},{{1}}}});


        output = conv2d_NHWC(input, kernel);
        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{28}},
                                    {{52}}}});
        THEN("output should be like expected"){
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values-output).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
}

SCENARIO( "Test conv2d_NHWC_backprop_kernel", "[Math]"){
    WHEN("backpro gradient") {
        int N = 1, H = 3, W = 3, C = 1, kernel_h = 2, kernel_w = 2, kernel_ci = 1, kernel_co = 1, out_h = 2, out_w = 2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1}, {2}, {3}},
                          {{4}, {5}, {6}},
                          {{7}, {8}, {9}}}});

        grad.setValues({{{{1}, {2}},
                         {{3}, {4}}}});

        kernel = conv2d_NHWC_backprop_kernel(input, grad);

        Eigen::Tensor<float, 4, RowMajor> expected_values(kernel_h, kernel_w, kernel_ci, kernel_co);
        expected_values.setValues({{{{37}}, {{47}}},
                                   {{{67}}, {{77}}}});

        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - kernel).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient with batch == 2") {
        int N = 2, H = 3, W = 3, C = 1, kernel_h = 2, kernel_w = 2, kernel_ci = 1, kernel_co = 1, out_h = 2, out_w = 2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1},{2},{3}},
                          {{4},{5},{6}},
                          {{7},{8},{9}}},
                         {{{2},{2},{2}},
                          {{4},{4},{4}},
                          {{6},{6},{6}}}
                        });

        grad.setValues({{{{1}, {2}},
                         {{3}, {4}}},
                        {{{2}, {4}},
                         {{6}, {8}}}});

        kernel = conv2d_NHWC_backprop_kernel(input, grad);

        Eigen::Tensor<float, 4, RowMajor> expected_values(kernel_h, kernel_w, kernel_ci, kernel_co);
        expected_values.setValues({{{{105}}, {{115}}},
                                   {{{175}}, {{185}}}});

        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - kernel).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient input channel == 2") {
        int N = 1, H = 3, W = 3, C = 2, kernel_h = 2, kernel_w = 2, kernel_ci = 2, kernel_co = 1, out_h = 2, out_w = 2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1,1},{2,2},{3,3}},
                          {{4,4},{5,5},{6,6}},
                          {{7,7},{8,8},{9,9}}}});

        grad.setValues({{{{1}, {2}},
                         {{3}, {4}}}});

        kernel = conv2d_NHWC_backprop_kernel(input, grad);

        Eigen::Tensor<float, 4, RowMajor> expected_values(kernel_h, kernel_w, kernel_ci, kernel_co);
        expected_values.setValues({{{{37},{37}}, {{47},{47}}},
                                   {{{67},{67}}, {{77},{77}}}});

        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - kernel).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient output channel == 3") {
        int N = 1, H = 3, W = 3, C = 1, kernel_h = 2, kernel_w = 2, kernel_ci = 1, kernel_co = 3, out_h = 2, out_w = 2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1}, {2}, {3}},
                          {{4}, {5}, {6}},
                          {{7}, {8}, {9}}}});

        grad.setValues({{{{1,2,3}, {2,3,4}},
                         {{3,4,5}, {4,5,6}}}});

        kernel = conv2d_NHWC_backprop_kernel(input, grad);

        Eigen::Tensor<float, 4, RowMajor> expected_values(kernel_h, kernel_w, kernel_ci, kernel_co);
        expected_values.setValues({{{{37,49,61}}, {{47,63,79}}},
                                   {{{67,91,115}}, {{77,105,133}}}});

        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - kernel).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient with batch == 2 channel_input==2 channel_output==3") {
        int N = 2, H = 3, W = 3, C = 2, kernel_h = 2, kernel_w = 2, kernel_ci = 2, kernel_co = 3, out_h = 2, out_w = 2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1,1},{2,2},{3,3}},
                          {{4,4},{5,5},{6,6}},
                          {{7,7},{8,8},{9,9}}},
                         {{{2,2},{2,2},{2,2}},
                          {{4,4},{4,4},{4,4}},
                          {{6,6},{6,6},{6,6}}}
                        });

        grad.setValues({{{{1,2,3}, {2,3,4}},
                         {{3,4,5}, {4,5,6}}},
                        {{{2,2,2}, {4,4,4}},
                         {{6,6,6}, {8,8,8}}}});

        kernel = conv2d_NHWC_backprop_kernel(input, grad);

        Eigen::Tensor<float, 4, RowMajor> expected_values(kernel_h, kernel_w, kernel_ci, kernel_co);
        expected_values.setValues({{{{105,117,129},{105,117,129}}, {{115,131,147},{115,131,147}}},
                                   {{{175,199,223},{175,199,223}}, {{185,213,241},{185,213,241}}}});

        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - kernel).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }

    WHEN("backpro gradient with asymetric kernel") {
        int N = 1, H = 4, W = 3, C=1, kernel_h = 3, kernel_w = 3, kernel_ci=1,kernel_co=1, out_h=2, out_w=1;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        input.setValues({{{{1}, {2}, {3}},
                          {{4}, {5}, {6}},
                          {{7}, {8}, {9}},
                          {{7}, {8}, {9}}}});

        grad.setValues({{{{1}},
                         {{2}}}});

        kernel = conv2d_NHWC_backprop_kernel(input, grad);

        Eigen::Tensor<float, 4, RowMajor> expected_values(kernel_h, kernel_w, kernel_ci, kernel_co);
        expected_values.setValues({{{{ 9}}, {{12}}, {{15}}},
                                   {{{18}}, {{21}}, {{24}}},
                                   {{{21}}, {{24}}, {{27}}}});


        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - kernel).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
}

SCENARIO( "Test conv2d_NHWC_backprop_input", "[Math]"){

    WHEN("backpro gradient with simple kernel") {
        int N = 1, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        kernel.setValues({{{{1}},{{2}}},
                          {{{3}},{{4}}}});

        grad.setValues({{{{2}, {4}},
                         {{6}, {8}}}});

        input = conv2d_NHWC_backprop_input(grad, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, H, W, C);
        expected_values.setValues({{{{ 2}, { 8}, { 8}},
                                    {{12}, {40}, {32}},
                                    {{18}, {48}, {32}}}});


        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - input).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient with asymetric kernel") {
        int N = 1, H = 3, W = 4, C=1, kernel_h = 2, kernel_w = 3, kernel_ci=1,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        kernel.setValues({{{{1}},{{2}},{{3}}},
                          {{{4}},{{5}},{{6}}}});

        grad.setValues({{{{2}, {4}},
                         {{6}, {8}}}});

        input = conv2d_NHWC_backprop_input(grad, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, H, W, C);
        expected_values.setValues({{{{ 2}, { 8}, {14}, {12}},
                                    {{14}, {46}, {66}, {48}},
                                    {{24}, {62}, {76}, {48}}}});


        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - input).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient with ci == 2") {
        int N = 1, H = 3, W = 3, C=2, kernel_h = 2, kernel_w = 2, kernel_ci=2,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        kernel.setValues({{{{1},{2}},{{2},{4}}},
                          {{{3},{6}},{{4},{8}}}});

        grad.setValues({{{{2}, {4}},
                         {{6}, {8}}}});

        input = conv2d_NHWC_backprop_input(grad, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, H, W, C);
        expected_values.setValues({{{{ 2, 4}, { 8,16}, { 8,16}},
                                    {{12,24}, {40,80}, {32,64}},
                                    {{18,36}, {48,96}, {32,64}}}});


        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - input).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient with kernel_co == 2") {
        int N = 1, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=2, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        kernel.setValues({{{{1,2}},{{2,4}}},
                          {{{3,6}},{{4,8}}}});

        grad.setValues({{{{2,3}, {3,4}},
                         {{4,5}, {5,6}}}});

        input = conv2d_NHWC_backprop_input(grad, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, H, W, C);
        expected_values.setValues({{{{ 8}, { 27}, {22}},
                                    {{38}, {110}, {78}},
                                    {{42}, {107}, {68}}}});


        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - input).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }

    WHEN("backpro gradient with batch == 2") {
        int N = 2, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        kernel.setValues({{{{1}},{{2}}},
                          {{{3}},{{4}}}});

        grad.setValues({{{{2}, {4}},
                         {{6}, {8}}},
                        {{{1}, {2}},
                         {{3}, {4}}}});

        input = conv2d_NHWC_backprop_input(grad, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, H, W, C);
        expected_values.setValues({{{{ 2}, { 8}, { 8}},
                                    {{12}, {40}, {32}},
                                    {{18}, {48}, {32}}},
                                   {{{ 1}, { 4}, { 4}},
                                    {{ 6}, {20}, {16}},
                                    {{ 9}, {24}, {16}}}});


        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - input).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
    WHEN("backpro gradient batch == 2, kernel_ci ==2, kernel_co ==2") {
        int N = 2, H = 3, W = 3, C=2, kernel_h = 2, kernel_w = 2, kernel_ci=2,kernel_co=2, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N, H, W, C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);
        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c

        kernel.setValues({{{{2,3},{2,3}},{{4, 6},{4, 6}}},
                          {{{6,8},{6,8}},{{8,10},{8,10}}}});
        kernel.setValues({{{{2,2},{3,3}},{{4, 4},{6, 6}}},
                          {{{6,6},{8,8}},{{8, 8},{10,10}}}});

        grad.setValues({{{{1,2}, {2,3}},
                         {{3,4}, {4,5}}},
                        {{{1,2}, {2,3}},
                         {{3,4}, {4,5}}}});

        input = conv2d_NHWC_backprop_input(grad, kernel);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, H, W, C);
        expected_values.setValues({{{{ 6, 9}, { 22, 33}, {20, 30}},
                                    {{32,45}, {100,139}, {76,104}},
                                    {{42,56}, {110,142}, {72, 90}}},
                                   {{{ 6, 9}, { 22, 33}, {20, 30}},
                                    {{32,45}, {100,139}, {76,104}},
                                    {{42,56}, {110,142}, {72, 90}}}});


        THEN("output should be like expected") {
            Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - input).abs().maximum();
            CHECK(ret_val(0) == 0);
        }
    }
}