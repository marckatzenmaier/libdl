//
// Created by marc on 06.05.19.
//
#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>

/*SCENARIO( "", "[]"){
    GIVEN(""){
        WHEN(""){
            THEN(""){
                REQUIRE(true == true);
            }
        }
    }
}*/

SCENARIO( "Test Graph Node basics", "[Node]"){
    GIVEN("name and data"){

        string name = "test";
        MatrixXf data = Eigen::MatrixXf(3,4);
        data(0,0) = 1;
        MatrixXf grad_wrong_size = Eigen::MatrixXf(1,2);
        grad_wrong_size(0,0) = 2;
        MatrixXf data_different_size = Eigen::MatrixXf(2,3);
        data_different_size(0,0) = 3;
        MatrixXf grad_right_size = Eigen::MatrixXf(3,4);
        grad_right_size(0,0) = 4;
        MatrixXf data_right_size = Eigen::MatrixXf(3,4);
        data_right_size(0,0) = 5;

        WHEN("Call Node constructed only with name"){
            GraphNode node = GraphNode(name);
            THEN("Name should be this name and data_tensor initialised with dynamic matrix"){
                CHECK(name == node.getName());
            }
        }
        WHEN("Call Node constructed with name and data"){
            GraphNode node = GraphNode(name, data);
            THEN("Name should be this name and data should match"){
                CHECK(name == node.getName());
                CHECK(node.getData().rows() == 3);
                CHECK(node.getData().cols() == 4);
                CHECK(node.getData()(0,0) == 1);
                CHECK(node.getGradient().rows() == 3);
                CHECK(node.getGradient().cols() == 4);
            }
        }
        WHEN("setting and getting data"){
            GraphNode node = GraphNode(name, data);
            node.setData(data_right_size);
            THEN("data should be set properly and throw an error if size changes"){
                CHECK(node.getData().rows() == 3);
                CHECK(node.getData().cols() == 4);
                CHECK(node.getData()(0,0) == 5);
                CHECK_THROWS_WITH(node.setData(data_different_size), Catch::Contains( "error data size changing forbidden"));
            }
        }
        WHEN("setting data with different sice and previous size was 0,0"){
            GraphNode node = GraphNode(name, MatrixXf());
            THEN("data should be set properly and throw no error"){
                CHECK_NOTHROW(node.setData(data_right_size));
                CHECK(node.getData().rows() == 3);
                CHECK(node.getData().cols() == 4);
                CHECK(node.getData()(0,0) == 5);
            }
        }
        WHEN("setting and getting data"){
            GraphNode node = GraphNode(name, data);
            node.setGradient(grad_right_size);
            THEN("data should be set properly and throw an error if size changes"){
                CHECK(node.getGradient().rows() == 3);
                CHECK(node.getGradient().cols() == 4);
                CHECK(node.getGradient()(0,0) == 4);
                CHECK_THROWS_WITH(node.setGradient(grad_wrong_size), Catch::Contains( "error gradient size miss match"));
            }
        }

    }
}