//
// Created by marc on 17.05.19.
//

#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/variable.h"
#include "libdl/initializer.h"

using namespace Catch::literals;

using namespace std;
using namespace Eigen;

SCENARIO( "Test initialisation", "[init]"){
    GIVEN("variable with huge dimension"){

        Tensor4f data1 = Tensor4f(200,1,1,300);
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data1));
        WHEN("data initialised"){
            init_random(var1);
            THEN("mean should equal zero and standard deviation should be equal 1/tensorsize"){
                Eigen::Tensor<float, 0, RowMajor> mean = var1->getData().mean();
                Eigen::Tensor<float, 4, RowMajor> mean_tensor(200,1,1,300);
                mean_tensor.setConstant(mean(0));
                Eigen::Tensor<float, 0, RowMajor> var =(var1->getData()-mean_tensor).square().mean();
                CHECK(mean(0) < 0.001);
                CHECK(mean(0) > -0.001);
                CHECK(var(0) < 0.0055);
                CHECK(var(0) > 0.0045);
            }
        }
    }
    GIVEN("variable with huge dimension"){

        Tensor4f data1 = Tensor4f(200,1,1,300);
        shared_ptr<Variable> var1 = make_shared<Variable>(Variable("1", data1));
        Tensor4f data2 = Tensor4f(200,1,1,300);
        shared_ptr<Variable> var2 = make_shared<Variable>(Variable("1", data2));
        WHEN("data initialised"){
            init_weights_random(vector<shared_ptr<Variable>>({var1, var2}));
            THEN("mean of var1 should equal zero and standard deviation should be equal 1/tensorsize"){
                Eigen::Tensor<float, 0, RowMajor> mean = var1->getData().mean();
                Eigen::Tensor<float, 4, RowMajor> mean_tensor(200,1,1,300);
                mean_tensor.setConstant(mean(0));
                Eigen::Tensor<float, 0, RowMajor> var =(var1->getData()-mean_tensor).square().mean();
                CHECK(mean(0) < 0.001);
                CHECK(mean(0) > -0.001);
                CHECK(var(0) < 0.0055);
                CHECK(var(0) > 0.0045);
            }
            THEN("mean of var 2 should equal zero and standard deviation should be equal 1/tensorsize"){
                Eigen::Tensor<float, 0, RowMajor> mean = var2->getData().mean();
                Eigen::Tensor<float, 4, RowMajor> mean_tensor(200,1,1,300);
                mean_tensor.setConstant(mean(0));
                Eigen::Tensor<float, 0, RowMajor> var =(var2->getData()-mean_tensor).square().mean();
                CHECK(mean(0) < 0.001);
                CHECK(mean(0) > -0.001);
                CHECK(var(0) < 0.0055);
                CHECK(var(0) > 0.0045);
            }
        }
    }
}