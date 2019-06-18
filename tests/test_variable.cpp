//
// Created by marc on 09.05.19.
//


#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"

using namespace std;
using namespace Eigen;

SCENARIO( "Test Variable", "[Node]"){
    GIVEN("Variable Node, data, gradient"){

        string name = "test_var";
        Tensor4f data = Tensor4f(1,2,3,1);
        data.setValues({{{{1},{2},{3}},{{4},{5},{6}}}});
        Tensor4f grad = Tensor4f(1,2,3,1);
        grad.setValues({{{{7},{8},{9}},{{10},{11},{12}}}});// 7, 8, 9, 10, 11, 12;
        Variable var = Variable(name, data);
        var.setGradient(grad);
        WHEN("Forward called"){
            var.forward();
            THEN(" data and gradient doesn't change"){
                Eigen::Tensor<float, 0, RowMajor> val = (var.getData() - data).abs().maximum();
                REQUIRE(val(0) == 0);
                val = (var.getGradient() - grad).abs().maximum();
                REQUIRE(val(0) == 0);
            }
        }
        WHEN("Backward called"){
            var.backward();
            THEN("data and gradient doesn't change"){
                Eigen::Tensor<float, 0, RowMajor> val = (var.getData() - data).abs().maximum();
                REQUIRE(val(0) == 0);
                val = (var.getGradient() - grad).abs().maximum();
                REQUIRE(val(0) == 0);
            }
        }

    }
}