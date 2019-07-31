//
// Created by marc on 06.05.19.
//
#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include "libdl/variable.h"
#include <iostream>
#include "libdl/placeholder.h"

using namespace std;
using namespace Eigen;

SCENARIO( "Test Graph Node basics based on the simple Variable class", "[Node]"){
    GIVEN("name and data"){

        string name = "test";
        Tensor4f data = Tensor4f(1,2,3,4);
        data(0,0,0,0) = 1;
        Tensor4f grad_wrong_size = Tensor4f(1,3,2,5);
        Tensor4f data_different_size = Tensor4f(2,3,4,5);
        Tensor4f grad_right_size = Tensor4f(1,2,3,4);
        grad_right_size(0,0,0,0) = 4;
        Tensor4f data_right_size = Tensor4f(1,2,3,4);
        data_right_size(0,0,0,0) = 5;
        Tensor4f data_different_batch_size = Tensor4f(5,2,3,4);
        WHEN("Call Node constructed only with name"){
            Variable node = Variable(name,data);
            THEN("Name should be this name"){
                CHECK("Variable" == node.getType());
            }
        }
        WHEN("Call Placeholder Node constructed with zero size tensor"){
            //Placeholder node = Placeholder(name,Tensor4f(0,0,0,0));

            THEN("Setting grad should cause no error"){
                Placeholder node = Placeholder(name,Tensor4f(1,1,1,0));
                node.setData(data);
                CHECK_NOTHROW(node.setGradient(data));
            }
            THEN("Setting gard should cause no error"){
                Placeholder node = Placeholder(name,Tensor4f(1,1,0,1));
                node.setData(data);
                CHECK_NOTHROW(node.setGradient(data));
            }
            THEN("Setting gard should cause no error"){
                Placeholder node = Placeholder(name,Tensor4f(1,0,1,1));
                node.setData(data);
                CHECK_NOTHROW(node.setGradient(data));
            }
            THEN("Setting gard should cause no error"){
                Placeholder node = Placeholder(name,Tensor4f(0,1,1,1));
                node.setData(data);
                CHECK_NOTHROW(node.setGradient(data));
            }
            THEN("Setting data should cause error"){
                Placeholder node = Placeholder(name,Tensor4f(1,1,1,1));
                node.setGradient(Tensor4f(1,1,1,1));
                CHECK_THROWS_WITH(node.setGradient(Tensor4f(2,2,2,2)), Catch::Contains( "error gradient size miss match in node"));
            }
        }
        WHEN("Call Node constructed only with name"){
            Variable node = Variable(name,data);
            THEN("Name should be this name"){
                CHECK(name == node.getName());
            }
        }
        WHEN("setting and getting data"){
            Variable node = Variable(name, data_right_size);
            //node.setData(data_right_size);
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
            Variable node = Variable(name, data_right_size);
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
            Variable node = Variable(name, data_right_size);
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
            Variable node = Variable(name, data_right_size);
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