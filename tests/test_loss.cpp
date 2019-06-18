//
// Created by marc on 17.05.19.
//

#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"
#include "libdl/loss.h"

using namespace std;
using namespace Eigen;

SCENARIO( "Test loss_MSE", "[loss]"){
    GIVEN("prediction and label"){

        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,0,0.5}}},{{{2,0,-1}}}});
        shared_ptr<GraphNode> var1 = make_shared<GraphNode>(Variable("1", data1));
        Tensor4f data2 = Tensor4f(2,1,1,3);
        data2.setValues({{{{1,0,0.5}}},{{{2,0,-1}}}});
        shared_ptr<GraphNode> label1 = make_shared<GraphNode>(Variable("1", data2));
        Tensor4f data3 = Tensor4f(2,1,1,3);
        data3.setValues({{{{-0.5,1.5,-1}}},{{{0.5,3,0.5}}}});
        shared_ptr<GraphNode> label2 = make_shared<GraphNode>(Variable("1", data3));

        WHEN("data and label are equal"){
            var1->clearGradient();
            float result = loss_MSE(var1, label1);
            Eigen::Tensor<float, 0, RowMajor> result_grad = var1->getGradient().abs().maximum();
            THEN(" loss should be 0"){
                CHECK(result == 0);
                CHECK(result_grad(0)==0);
            }
        }
        WHEN("data and label are not equal"){

            Tensor4f data_expected = Tensor4f(2,1,1,3);
            data_expected.setValues({{{{0.5,-0.5,0.5}}},{{{0.5,-1,-0.5}}}});
            var1->clearGradient();
            float result = loss_MSE(var1, label2);
            Eigen::Tensor<float, 0, RowMajor> result_grad = (var1->getGradient()-data_expected).abs().maximum();
            THEN(" loss should be the expected value"){
                CHECK(result == 3.375);
                CHECK(result_grad(0)==0);
            }
        }

    }
}