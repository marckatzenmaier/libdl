//
// Created by marc on 06.05.19.
//
#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>

using namespace std;
using namespace Eigen;

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
        Tensor4f data = Tensor4f(1,2,3,4);
        data(0,0,0,0) = 1;
        Tensor4f grad_wrong_size = Tensor4f(1,2,3,5);
        Tensor4f data_different_size = Tensor4f(2,3,4,5);
        Tensor4f grad_right_size = Tensor4f(1,2,3,4);
        grad_right_size(0,0,0,0) = 4;
        Tensor4f data_right_size = Tensor4f(1,2,3,4);
        data_right_size(0,0,0,0) = 5;
        Tensor4f data_different_batch_size = Tensor4f(5,2,3,4);

        WHEN("Call Node constructed only with name"){
            GraphNode node = GraphNode(name);
            THEN("Name should be this name and data_tensor initialised with dynamic matrix"){
                CHECK(name == node.getName());
            }
        }
        WHEN("setting and getting data"){
            GraphNode node = GraphNode(name);
            node.setData(data_right_size);
            THEN("data should be set properly and throw an error if size changes"){
                CHECK(node.getData().dimension(0) == 1);
                CHECK(node.getData().dimension(1) == 2);
                CHECK(node.getData().dimension(2) == 3);
                CHECK(node.getData().dimension(3) == 4);
                CHECK(node.getData()(0,0,0,0) == 5);
                CHECK_THROWS_WITH(node.setData(data_different_size), Catch::Contains( "error data size changing forbidden"));
                CHECK_NOTHROW(node.setData(data_different_batch_size));
            }
        }
        WHEN("setting and getting gradient"){
            GraphNode node = GraphNode(name);
            node.setData(data_right_size);
            node.setGradient(grad_right_size);
            THEN("gradient should be set properly and throw an error if size changes"){
                CHECK(node.getGradient().dimension(0) == 1);
                CHECK(node.getGradient().dimension(1) == 2);
                CHECK(node.getGradient().dimension(2) == 3);
                CHECK(node.getGradient().dimension(3) == 4);
                CHECK(node.getGradient()(0,0,0,0) == 4);
                CHECK_THROWS_WITH(node.setGradient(grad_wrong_size), Catch::Contains( "error gradient size miss match"));
            }
        }
        WHEN("clear gradient"){
            GraphNode node = GraphNode(name);
            node.setData(data_right_size);
            node.setGradient(grad_right_size);
            node.clearGradient();
            THEN("gradient should be set to 0"){
                CHECK(node.getGradient().dimension(0) == 1);
                CHECK(node.getGradient().dimension(1) == 2);
                CHECK(node.getGradient().dimension(2) == 3);
                CHECK(node.getGradient().dimension(3) == 4);
                CHECK(node.getGradient()(0,0,0,0) == 0);
            }
        }
        WHEN("add gradient"){
            GraphNode node = GraphNode(name);
            node.setData(data_right_size);
            node.setGradient(grad_right_size);
            node.addGradient(grad_right_size);
            THEN("gradient should be set to 0"){
                CHECK(node.getGradient().dimension(0) == 1);
                CHECK(node.getGradient().dimension(1) == 2);
                CHECK(node.getGradient().dimension(2) == 3);
                CHECK(node.getGradient().dimension(3) == 4);
                CHECK(node.getGradient()(0,0,0,0) == 8);
            }
        }

    }
}