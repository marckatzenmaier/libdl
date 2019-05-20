//
// Created by marc on 17.05.19.
//
#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include "libdl/Optimizer.h"

using namespace std;
using namespace Eigen;
SCENARIO( "Test Optimizer", "[Node]"){
    GIVEN("Basic Nodes which forms a simple matrix multiplication graph"){
        string name = "test_opp";
        MatrixXf data1 = Eigen::MatrixXf(2,3);
        data1 << 1, 2, 3, 4, 5, 6;
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data1));

        MatrixXf data2 = Eigen::MatrixXf(3,2);
        data2 << 1, 2, 3, 4, 5, 6;
        shared_ptr<GraphNode> var2 = make_shared<Variable>(Variable("2", data2));


        MatrixXf data3 = Eigen::MatrixXf(2,2);
        data3 << 1, 2, 3, 4;
        shared_ptr<GraphNode> var3 = make_shared<Variable>(Variable("3", data3));

        NodeVec input_nodes_1 = {var1, var2};
        MatrixMultiplication opp_obj = MatrixMultiplication("MatMul", input_nodes_1);
        shared_ptr<GraphNode> opp = make_shared<MatrixMultiplication>(opp_obj);

        NodeVec input_nodes_2 = {opp, var3};
        ElementwiseAdd opp2_obj = ElementwiseAdd("EleAdd", input_nodes_2);
        shared_ptr<GraphNode> opp2 = make_shared<ElementwiseAdd>(opp2_obj);
        WHEN("Constructor called"){
            Graph graph = Graph(opp2);
            graph.forward();
            graph.backward();
            THEN("calc_forward_order and calc_backward_order need to be calculated"){
            REQUIRE(true); //todo figure out if this is testable e.g. no exception
            }
        }
    }
}