//
// Created by marc on 09.05.19.
//


#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"

SCENARIO( "Test Variable", "[Node]"){
    GIVEN("Variable Node, data, gradient"){

        string name = "test_var";
        MatrixXf data = Eigen::MatrixXf(2,3);
        data << 1, 2, 3, 4, 5, 6;
        MatrixXf grad = Eigen::MatrixXf(2,3);
        grad << 7, 8, 9, 10, 11, 12;
        Variable var = Variable(name, data);
        var.setGradient(grad);
        WHEN("Forward called"){
            var.forward();
            THEN(" data and gradient doesn't change"){
                REQUIRE((var.getData() - data).norm() == 0);
                REQUIRE((var.getGradient() - grad).norm() == 0);
            }
        }
        WHEN("Backward called"){
            var.backward();
            THEN("data and gradient doesn't change"){
                REQUIRE((var.getData() - data).norm() == 0);
                REQUIRE((var.getGradient() - grad).norm() == 0);
            }
        }

    }
}