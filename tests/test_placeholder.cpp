//
// Created by marc on 09.05.19.
//

#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include "libdl/placeholder.h"
#include "libdl/graph.h"
#include <iostream>
#include "Eigen/Core"
using namespace std;
using namespace Eigen;
SCENARIO( "Test Placeholder", "[Node]"){
    GIVEN("Placeholder Node, data, gradient"){

        string name = "test_var";
        Tensor4f data = Tensor4f(1,2,3,1);
        data.setValues({{{{1},{2},{3}},{{4},{5},{6}}}});
        Tensor4f grad = Tensor4f(1,2,3,1);
        grad.setValues({{{{7},{8},{9}},{{10},{11},{12}}}});
        Placeholder placeholder = Placeholder(name, data);
        placeholder.setGradient(grad);
        CHECK(placeholder.getType()=="Placeholder");
        Graph g(make_shared<Placeholder>(placeholder));
        WHEN("Forward called"){
            g.forward();
            THEN(" data and gradient doesn't change"){
                Eigen::Tensor<float, 0, RowMajor> val = (placeholder.getData() - data).abs().maximum();
                REQUIRE(val(0) == 0);
                val = (placeholder.getGradient() - grad).abs().maximum();
                REQUIRE(val(0) == 0);
            }
        }
        WHEN("Backward called"){
            g.backward();
            THEN("data and gradient doesn't change"){
                Eigen::Tensor<float, 0, RowMajor> val = (placeholder.getData() - data).abs().maximum();
                REQUIRE(val(0) == 0);
                val = (placeholder.getGradient() - grad).abs().maximum();
                REQUIRE(val(0) == 0);
            }
        }

    }
}