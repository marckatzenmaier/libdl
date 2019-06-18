//
// Created by marc on 09.05.19.
//

#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include <libdl/Optimizer.h>
#include "libdl/initializer.h"

using namespace std;
using namespace Eigen;

SCENARIO( "Test Graph", "[Node]"){
    GIVEN("Basic Nodes which forms a simple matrix multiplication graph"){
        string name = "test_opp";
        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,2,3}}},{{{4,5,6}}}});
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data1));

        Tensor4f data2 = Tensor4f(1,1,3,2);
        data2.setValues({{{{1,2},{3,4},{5,6}}}});
        shared_ptr<GraphNode> var2 = make_shared<Variable>(Variable("2", data2));


        Tensor4f data3 = Tensor4f(2,1,1,2);
        data3.setValues({{{{1,2},{3,4}}}});
        shared_ptr<GraphNode> var3 = make_shared<Variable>(Variable("3", data3));

        NodeVec input_nodes_1 = {var1, var2};
        MatrixMultiplication opp_obj = MatrixMultiplication("MatMul", input_nodes_1);
        shared_ptr<GraphNode> opp = make_shared<MatrixMultiplication>(opp_obj);

        NodeVec input_nodes_2 = {opp, var3};
        ElementwiseAdd opp2_obj = ElementwiseAdd("EleAdd", input_nodes_2);
        shared_ptr<GraphNode> opp2 = make_shared<ElementwiseAdd>(opp2_obj);
        WHEN("forward called"){
            Graph graph = Graph(opp2);
            graph.forward();

            THEN("data should be calculated correct"){
                CHECK(opp->getData()(0,0,0,0)==22);
                CHECK(opp->getData()(0,0,0,1)==28);
                CHECK(opp->getData()(1,0,0,0)==49);
                CHECK(opp->getData()(1,0,0,1)==64);
                CHECK(opp2->getData()(0,0,0,0)==23);
                CHECK(opp2->getData()(0,0,0,1)==30);
                CHECK(opp2->getData()(1,0,0,0)==52);
                CHECK(opp2->getData()(1,0,0,1)==68);
            }
        }
        WHEN("forward and backward called"){
            Graph graph = Graph(opp2);
            graph.forward();
            Tensor4f grad = Tensor4f(2,1,1,2);
            grad.setValues({{{{1,2},{3,4}}}});
            opp2->setGradient(grad);
            graph.backward();

            THEN("gradient schould be populated correct"){
                CHECK(var1->getGradient()(0,0,0,0)==5);
                CHECK(var1->getGradient()(0,0,0,1)==11);
                CHECK(var1->getGradient()(0,0,0,2)==17);
                CHECK(var1->getGradient()(1,0,0,0)==11);
                CHECK(var1->getGradient()(1,0,0,1)==25);
                CHECK(var1->getGradient()(1,0,0,2)==39);

                CHECK(var2->getGradient()(0,0,0,0)==13);
                CHECK(var2->getGradient()(0,0,0,1)==18);
                CHECK(var2->getGradient()(0,0,1,0)==17);
                CHECK(var2->getGradient()(0,0,1,1)==24);
                CHECK(var2->getGradient()(0,0,2,0)==21);
                CHECK(var2->getGradient()(0,0,2,1)==30);

                CHECK(var3->getGradient()(0,0,0,0)==1);
                CHECK(var3->getGradient()(0,0,0,1)==2);
                CHECK(var3->getGradient()(1,0,0,0)==3);
                CHECK(var3->getGradient()(1,0,0,1)==4);
            }
        }
    }
}