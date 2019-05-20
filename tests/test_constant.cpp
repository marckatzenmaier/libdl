//
// Created by marc on 09.05.19.
//

#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include "libdl/Constant.h"
#include <iostream>
/*SCENARIO( "Test Constant", "[Node]"){
    GIVEN("Name, Values in type Tensor, Matrix, float"){
        Tensor tensor;
        tensor.data = Eigen::MatrixXf(3,4);
        tensor.data(0,0) = 1;
        tensor.gradient = Eigen::MatrixXf(1,2);
        tensor.gradient(0,0) = 3;
        Eigen::MatrixXf matrix = Eigen::MatrixXf(2,3);
        matrix(0,0) = 3;
        float scalar = 2.0;
        string name = "test";
        WHEN("constructed with name"){
            Constant constant = Constant(tensor, name);
            Constant constant1 = Constant(matrix, name);
            Constant constant2 = Constant(1.0, 0, 0, name);
            THEN("Node should have this name"){
                CHECK(constant.getName() == name);
                CHECK(constant1.getName() == name);
                CHECK(constant2.getName() == name);
            }
        }
        WHEN("constructed with Tensor"){
            Constant constant = Constant(tensor);
            THEN("Inner state should be populated correctly"){
                CHECK(constant.getName() != "");
                CHECK(constant.getInnerState().data(0,0) == 1);
                CHECK(constant.getInnerState().data.rows() == 3);
                CHECK(constant.getInnerState().data.cols() == 4);

            }
        }
        WHEN("constructed with Martix"){
            Constant constant = Constant(matrix);
            THEN("Inner state should be populated correctly"){
                CHECK(constant.getName() != "");
                CHECK(constant.getInnerState().data(0,0) == 3);
                CHECK(constant.getInnerState().data.rows() == 2);
                CHECK(constant.getInnerState().data.cols() == 3);
            }
        }
        WHEN("constructed with float"){
            Constant constant = Constant(scalar, 1,3);
            THEN("Inner state should be populated correctly"){
                CHECK(constant.getName() != "");
                CHECK(constant.getInnerState().data(0,0) == scalar);
                CHECK(constant.getInnerState().data.rows() == 1);
                CHECK(constant.getInnerState().data.cols() == 3);
            }
        }
        WHEN("forward pass"){
            Constant constant = Constant(scalar, 1,3);
            constant.forward();
            THEN("output should be inner state"){
                CHECK(constant.getName() != "");
                CHECK(constant.getInnerState().data(0,0) == scalar);
                CHECK(constant.getInnerState().data.rows() == 1);
                CHECK(constant.getInnerState().data.cols() == 3);
            }
        }
    }
}*/