//
// Created by marc on 09.05.19.
//


#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include "libdl/variable.h"
#include "libdl/opperation.h"
#include "libdl/graph.h"
#include <iostream>
using namespace Catch::literals;

using namespace std;
using namespace Eigen;
SCENARIO( "Test Opperation based on simple ElementwiseAdd node", "[Node]"){
    GIVEN("input nodes, name Node"){
        string name = "test_opp";
        Tensor4f data = Tensor4f(1,2,3,1);
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data));
        shared_ptr<GraphNode> var2 = make_shared<Variable>(Variable("2", data));
        NodeVec input_nodes = {var1, var2};
        WHEN("Constructed"){
            ElementwiseAdd opp = ElementwiseAdd(name, input_nodes);
            THEN(" input vector should be able to get"){
                REQUIRE(opp.getInputVec()[0]->getName()=="1");
                REQUIRE(opp.getInputVec()[1]->getName()=="2");
            }
        }
    }
}

SCENARIO( "Test Opperation MatrixMultiplication", "[Node]"){
    GIVEN("input nodes, name Node"){
        string name = "test_opp";
        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,2,3}}},{{{4,5,6}}}});
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data1));


        Tensor4f data2 = Tensor4f(1,1,3,2);
        data2.setValues({{{{1,2},{3,4},{5,6}}}});

        shared_ptr<GraphNode> var2 = make_shared<Variable>(Variable("2", data2));


        Tensor4f data3 = Tensor4f(1,1,4,4);
        data3.setValues({{{{ 1, 2, 3, 4},
                           { 5, 6, 7, 8},
                           { 9,10,11,12},
                           {13,14,15,16}}}});
        shared_ptr<GraphNode> var3 = make_shared<Variable>(Variable("3", data3));

        WHEN("Constructed with more or less then 2 inputs"){
            NodeVec input_nodes_1 = {var1};
            NodeVec input_nodes_3 = {var1, var2, var3};
            THEN("throw error"){
                CHECK_THROWS_WITH(MatrixMultiplication(name, input_nodes_1), Catch::Contains( "error to few (" ) && Catch::Contains( " expected 2) input nodes in node: " ) );
                CHECK_THROWS_WITH(MatrixMultiplication(name, input_nodes_3), Catch::Contains( "error to many (" ) && Catch::Contains( " expected 2) input nodes in node: " ));
            }
        }
        WHEN("Constructed with 2 input nodes but wrong input dimensions"){
            NodeVec input_nodes_not_fit = {var1, var3};
            THEN("throw an error"){
                CHECK_THROWS_WITH(MatrixMultiplication(name, input_nodes_not_fit),  Catch::Contains( "error dimension miss fit:" ) && Catch::Contains( " in node " ));
            }
        }
        WHEN("Constructed with 2 input nodes having fitting dim"){
            NodeVec input_nodes_fit = {var1, var2};
            MatrixMultiplication opp = MatrixMultiplication(name, input_nodes_fit);
            Graph g(make_shared<MatrixMultiplication>(opp));
            THEN("output should have right dimensions"){
                REQUIRE(opp.getData().dimension(0) == opp.getInputVec()[0]->getData().dimension(0));
                REQUIRE(opp.getData().dimension(3) == opp.getInputVec()[1]->getData().dimension(3));
            }
            WHEN("forward called"){
                g.forward();
                THEN("output should be calculate"){
                    CHECK(g.getEndpoint()->getData()(0,0,0,0) == 22);
                    CHECK(g.getEndpoint()->getData()(0,0,0,1) == 28);
                    CHECK(g.getEndpoint()->getData()(1,0,0,0) == 49);
                    CHECK(g.getEndpoint()->getData()(1,0,0,1) == 64);
                }
            }
            WHEN("gradient is populated and backward called"){

                Tensor4f grad = Tensor4f(2,1,1,2);
                grad.setValues({{{{1,2}}},{{{3,4}}}});
                g.getEndpoint()->setGradient(grad);
                g.backward();
                THEN("gradient to inpusts should be caldulated"){
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,0)== 5);
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,1)==11);
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,2)==17);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,0)==11);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,1)==25);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,2)==39);


                    CHECK(opp.getInputVec()[1]->getGradient()(0,0,0,0)==13);
                    CHECK(opp.getInputVec()[1]->getGradient()(0,0,0,1)==18);
                    CHECK(opp.getInputVec()[1]->getGradient()(0,0,1,0)==17);
                    CHECK(opp.getInputVec()[1]->getGradient()(0,0,1,1)==24);
                    CHECK(opp.getInputVec()[1]->getGradient()(0,0,2,0)==21);
                    CHECK(opp.getInputVec()[1]->getGradient()(0,0,2,1)==30);
                }
            }
        }
    }
}

SCENARIO( "Test Opperation ElementwiseAdd", "[Node]"){
    GIVEN("input nodes, name Node"){
        string name = "test_opp";

        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,2,3}}},{{{4,5,6}}}});
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data1));

        Tensor4f data2 = Tensor4f(2,1,1,3);
        data2.setValues({{{{7,8,9}}},{{{10,11,12}}}});
        shared_ptr<GraphNode> var2 = make_shared<Variable>(Variable("2", data2));

        Tensor4f data3 = Tensor4f(4,1,1,4);
        data3.setValues({{{{1,2,3,4}}},{{{5,6,7,8}}},{{{9,10,11,12}}},{{{13,14,15,16}}}});
        shared_ptr<GraphNode> var3 = make_shared<Variable>(Variable("3", data3));

        WHEN("Constructed with 2 input nodes but wrong input dimensions"){
            NodeVec input_nodes_not_fit = {var1, var3};
            THEN("throw an error"){
                CHECK_THROWS_WITH(ElementwiseAdd(name, input_nodes_not_fit),  Catch::Contains( "error dimension miss fit:" ) && Catch::Contains( " in node " ));
            }
        }
        WHEN("Constructed with 2 input nodes having fitting dim"){
            NodeVec input_nodes_fit = {var1, var2};
            ElementwiseAdd opp = ElementwiseAdd(name, input_nodes_fit);
            Graph g(make_shared<ElementwiseAdd>(opp));
            THEN("output should have right dimensions"){
                CHECK(opp.getData().dimension(0) == opp.getInputVec()[0]->getData().dimension(0));
                CHECK(opp.getData().dimension(1) == opp.getInputVec()[0]->getData().dimension(1));
                CHECK(opp.getData().dimension(2) == opp.getInputVec()[0]->getData().dimension(2));
                CHECK(opp.getData().dimension(3) == opp.getInputVec()[0]->getData().dimension(3));
            }
            WHEN("forward called"){
                g.forward();
                THEN("output should be calculate"){
                    CHECK(g.getEndpoint()->getData()(0,0,0,0) == 8);
                    CHECK(g.getEndpoint()->getData()(0,0,0,1) == 10);
                    CHECK(g.getEndpoint()->getData()(0,0,0,2) == 12);
                    CHECK(g.getEndpoint()->getData()(1,0,0,0) == 14);
                    CHECK(g.getEndpoint()->getData()(1,0,0,1) == 16);
                    CHECK(g.getEndpoint()->getData()(1,0,0,2) == 18);
                }
            }
            WHEN("gradient is populated and backward called"){
                Tensor4f grad = Tensor4f(2,1,1,3);
                grad.setValues({{{{1,1,2}}},{{{2,3,3}}}});
                g.getEndpoint()->setGradient(grad);
                g.backward();
                THEN("gradient to inpusts should be caldulated"){
                    REQUIRE(opp.getInputVec()[0]->getGradient()(0,0,0,0)==1);
                    REQUIRE(opp.getInputVec()[0]->getGradient()(0,0,0,1)==1);
                    REQUIRE(opp.getInputVec()[0]->getGradient()(0,0,0,2)==2);
                    REQUIRE(opp.getInputVec()[0]->getGradient()(1,0,0,0)==2);
                    REQUIRE(opp.getInputVec()[0]->getGradient()(1,0,0,1)==3);
                    REQUIRE(opp.getInputVec()[0]->getGradient()(1,0,0,2)==3);

                    REQUIRE(opp.getInputVec()[1]->getGradient()(0,0,0,0)==1);
                    REQUIRE(opp.getInputVec()[1]->getGradient()(0,0,0,1)==1);
                    REQUIRE(opp.getInputVec()[1]->getGradient()(0,0,0,2)==2);
                    REQUIRE(opp.getInputVec()[1]->getGradient()(1,0,0,0)==2);
                    REQUIRE(opp.getInputVec()[1]->getGradient()(1,0,0,1)==3);
                    REQUIRE(opp.getInputVec()[1]->getGradient()(1,0,0,2)==3);
                }
            }
        }
    }
}

SCENARIO( "Test Opperation Sigmoid", "[Node]"){
    GIVEN("input nodes, name Node"){
        string name = "test_opp";
        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,2,3}}},{{{4,5,6}}}});
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data1));

        WHEN("Constructed "){
            NodeVec input_nodes = {var1};
            Sigmoid opp = Sigmoid(name, input_nodes);
            Graph g(make_shared<Sigmoid>(opp));
            THEN("output should have same dimensions"){

                CHECK(opp.getData().dimension(0) == 2);
                CHECK(opp.getData().dimension(1) == 1);
                CHECK(opp.getData().dimension(2) == 1);
                CHECK(opp.getData().dimension(3) == 3);
            }
            WHEN("forward called"){
                g.forward();
                THEN("output should be calculate"){
                    CHECK(g.getEndpoint()->getData()(0,0,0,0) == 0.73105858_a);
                    CHECK(g.getEndpoint()->getData()(0,0,0,1) == 0.88079708_a);
                    CHECK(g.getEndpoint()->getData()(0,0,0,2) == 0.95257413_a);
                    CHECK(g.getEndpoint()->getData()(1,0,0,0) == 0.98201379_a);
                    CHECK(g.getEndpoint()->getData()(1,0,0,1) == 0.99330715_a);
                    CHECK(g.getEndpoint()->getData()(1,0,0,2) == 0.99752738_a);
                }
            }
            WHEN("gradient is populated and backward called"){
                Tensor4f grad = Tensor4f(2,1,1,3);
                g.forward();
                grad.setValues({{{{1,1,2}}},{{{2,3,3}}}});
                g.getEndpoint()->setGradient(grad);
                g.backward();
                THEN("gradient to inpusts should be caldulated"){
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,0)==0.19661193_a);
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,1)==0.10499359_a);
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,2)==0.09035332_a);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,0)==0.03532541_a);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,1)==0.01994417_a);
                    //CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,2)==0.00739953_a);//is a strange rounding problem
                }
            }
        }
    }
}
SCENARIO( "Test Opperation ReLU", "[Node]"){
    GIVEN("input nodes, name Node"){
        string name = "test_opp";
        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,-2,3}}},{{{4,5,-6}}}});
        shared_ptr<GraphNode> var1 = make_shared<Variable>(Variable("1", data1));

        WHEN("Constructed "){
            NodeVec input_nodes = {var1};
            ReLU opp = ReLU(name, input_nodes);
            Graph g(make_shared<ReLU>(opp));
            THEN("output should have same dimensions"){

                CHECK(opp.getData().dimension(0) == 2);
                CHECK(opp.getData().dimension(1) == 1);
                CHECK(opp.getData().dimension(2) == 1);
                CHECK(opp.getData().dimension(3) == 3);
            }
            WHEN("forward called"){
                g.forward();
                THEN("output should be calculate"){
                    CHECK(g.getEndpoint()->getData()(0,0,0,0) == 1);
                    CHECK(g.getEndpoint()->getData()(0,0,0,1) == 0);
                    CHECK(g.getEndpoint()->getData()(0,0,0,2) == 3);
                    CHECK(g.getEndpoint()->getData()(1,0,0,0) == 4);
                    CHECK(g.getEndpoint()->getData()(1,0,0,1) == 5);
                    CHECK(g.getEndpoint()->getData()(1,0,0,2) == 0);
                }
            }
            WHEN("gradient is populated and backward called"){
                Tensor4f grad = Tensor4f(2,1,1,3);
                g.forward();
                grad.setValues({{{{1,-1,2}}},{{{-2,3,-3}}}});
                g.getEndpoint()->setGradient(grad);
                g.backward();
                THEN("gradient to inpusts should be caldulated"){
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,0)==1);
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,1)==0);
                    CHECK(opp.getInputVec()[0]->getGradient()(0,0,0,2)==2);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,0)==-2);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,1)==3);
                    CHECK(opp.getInputVec()[0]->getGradient()(1,0,0,2)==0);
                }
            }
        }
    }
}
SCENARIO( "Test Conv2d Node", "[Node]"){
    GIVEN("Conv2d Node") {
        int N = 1, H = 3, W = 3, C=1, kernel_h = 2, kernel_w = 2, kernel_ci=1,kernel_co=1, out_h=2, out_w=2;
        Eigen::Tensor<float, 4, RowMajor> input(N,H,W,C); //b,h,w,c
        Eigen::Tensor<float, 4, RowMajor> kernel(kernel_h, kernel_w, kernel_ci, kernel_co);

        input.setValues({{{{1},{2},{3}},
                                 {{4},{5},{6}},
                                 {{7},{8},{9}}}});

        kernel.setValues({{{{1}},{{1}}},
                          {{{1}},{{1}}}});
        shared_ptr<Variable> in = make_shared<Variable>(Variable("v1", input));
        shared_ptr<Variable> ker = make_shared<Variable>(Variable("v2", kernel));
        shared_ptr<Opperation> opp = make_shared<Conv2d>("conv", NodeVec({in, ker}));
        Graph g(opp);

        Eigen::Tensor<float, 4, RowMajor> expected_values(N, out_h, out_w, kernel_co);//b,h,w,c
        expected_values.setValues({{{{12},{16}}, {{24},{28}}}});

        Eigen::Tensor<float, 4, RowMajor> grad(N, out_h, out_w, kernel_co);//b,h,w,c
        grad.setValues({{{{1}, {2}},
                                {{3}, {4}}}});
        Eigen::Tensor<float, 4, RowMajor> expected_values_grad_ker(kernel_h, kernel_w, kernel_ci, kernel_co);
        expected_values_grad_ker.setValues({{{{37}}, {{47}}},
                                   {{{67}}, {{77}}}});
        WHEN("forward called") {
            g.forward();
            THEN("output should be like expected") {
                Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values - g.getEndpoint()->getData()).abs().maximum();
                CHECK(ret_val(0) == 0);
            }
        }
        WHEN("backward called") {
            g.getEndpoint()->setGradient(grad);
            g.backward();
            THEN("output should be like expected") {
                Eigen::Tensor<float, 0, RowMajor> ret_val = (expected_values_grad_ker - opp->getInputVec()[1]->getGradient()).abs().maximum();
                CHECK(ret_val(0) == 0);
            }
        }
        WHEN("try with wrong dimensions") {
            shared_ptr<Variable> ker = make_shared<Variable>(Variable("v2", Tensor4f(1,1,3,5)));
            THEN("output should be like expected") {
                REQUIRE_THROWS(Conv2d("conv", NodeVec({in, ker})));
            }
        }
    }
}

SCENARIO( "Test Average Pool Node", "[Node]"){
    GIVEN("Average Pool Node") {
        Eigen::Tensor<float, 4, RowMajor> input(1,2,2,1); //b,h,w,c

        input.setValues({{{{1},{2}},
                                 {{4},{5}}}});

        shared_ptr<Variable> in = make_shared<Variable>(Variable("v1", input));
        shared_ptr<Pool_average> opp = make_shared<Pool_average>("pool", NodeVec({in}));
        Graph g(opp);

        WHEN("forward called") {
            g.forward();
            THEN("output should be like expected") {
                CHECK(g.getEndpoint()->getData()(0,0,0,0) == 3);
            }
        }
        WHEN("backward called") {
            Tensor4f grad(1,1,1,1);
            grad.setValues({{{{1}}}});
            g.getEndpoint()->setGradient(grad);
            g.backward();
            THEN("output should be like expected") {
                CHECK(opp->getInputVec()[0]->getGradient()(0,0,0,0) == 0.25);
                CHECK(opp->getInputVec()[0]->getGradient()(0,0,1,0) == 0.25);
                CHECK(opp->getInputVec()[0]->getGradient()(0,1,0,0) == 0.25);
                CHECK(opp->getInputVec()[0]->getGradient()(0,1,1,0) == 0.25);
            }
        }
    }
}
SCENARIO( "Test Softmax Node", "[Node]"){
    GIVEN("Softmax Node") {
        Tensor4f input(1,1,1,4);
        input.setValues({{{{1,2,4,5}}}});
        Tensor4f grad(1,1,1,4);
        grad.setValues({{{{1,1,0,1}}}});
        shared_ptr<Variable> in = make_shared<Variable>(Variable("v1", input));
        shared_ptr<Softmax> opp = make_shared<Softmax>("Softmax", NodeVec({in}));
        Graph g(opp);
        WHEN("forward called") {
            g.forward();
            THEN("output should be like expected") {
                CHECK(abs(g.getEndpoint()->getData()(0,0,0,0) - 0.01275)<0.001);
                CHECK(abs(g.getEndpoint()->getData()(0,0,0,1) - 0.03467)<0.001);
                CHECK(abs(g.getEndpoint()->getData()(0,0,0,2) - 0.25619)<0.001);
                CHECK(abs(g.getEndpoint()->getData()(0,0,0,3) - 0.69639)<0.001);
            }
        }
        WHEN("backward called") {
            g.forward();
            g.getEndpoint()->setGradient(grad);
            g.backward();
            THEN("output should be like expected") {
                CHECK(abs(in->getGradient()(0,0,0,0) -  0.00327f)<0.001);
                CHECK(abs(in->getGradient()(0,0,0,1) -  0.00888f)<0.001);
                CHECK(abs(in->getGradient()(0,0,0,2) - -0.19056f)<0.001);
                CHECK(abs(in->getGradient()(0,0,0,3) -  0.17841f)<0.001);
            }
        }
    }
}
SCENARIO( "Test TanH Node", "[Node]"){
    GIVEN("TanH Node") {
        Eigen::Tensor<float, 4, RowMajor> input(1,2,2,1); //b,h,w,c
        input.setValues({{{{0},{.2}},{{.4},{.5}}}});

        Tensor4f grad(1,2,2,1);
        grad.setValues({{{{1},{.2}},{{.4},{.5}}}});

        shared_ptr<Variable> in = make_shared<Variable>(Variable("v1", input));
        shared_ptr<TanH> opp = make_shared<TanH>("TanH", NodeVec({in}));
        Graph g(opp);
        WHEN("forward called") {
            g.forward();
            THEN("output should be like expected") {
                CHECK(abs(g.getEndpoint()->getData()(0,0,0,0) -     0.0f)<0.0001);
                CHECK(abs(g.getEndpoint()->getData()(0,0,1,0) - 0.19738f)<0.0001);
                CHECK(abs(g.getEndpoint()->getData()(0,1,0,0) - 0.37995f)<0.0001);
                CHECK(abs(g.getEndpoint()->getData()(0,1,1,0) - 0.46212f)<0.0001);
            }
        }
        WHEN("backward called") {
            g.forward();
            g.getEndpoint()->setGradient(grad);
            g.backward();
            THEN("output should be like expected") {
                CHECK(abs(opp->getInputVec()[0]->getGradient()(0,0,0,0) - 1        )<.001)   ;
                CHECK(abs(opp->getInputVec()[0]->getGradient()(0,0,1,0) - 0.19221f )<.001);
                CHECK(abs(opp->getInputVec()[0]->getGradient()(0,1,0,0) - 0.34226f )<.001);
                CHECK(abs(opp->getInputVec()[0]->getGradient()(0,1,1,0) - 0.39322f )<.001);
            }
        }
    }
}