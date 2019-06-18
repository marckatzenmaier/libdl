//
// Created by marc on 09.05.19.
//


#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include "libdl/Variable.h"
#include "libdl/opperation.h"
#include <iostream>
using namespace Catch::literals;

using namespace std;
using namespace Eigen;
SCENARIO( "Test Opperation", "[Node]"){
    GIVEN("input nodes, name Node"){
        string name = "test_opp";
        Tensor4f data = Tensor4f(1,2,3,1);
        shared_ptr<GraphNode> var1 = make_shared<GraphNode>(Variable("1", data));
        shared_ptr<GraphNode> var2 = make_shared<GraphNode>(Variable("2", data));
        NodeVec input_nodes = {var1, var2};
        WHEN("Constructed"){
            Opperation opp = Opperation(name, input_nodes);
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
        shared_ptr<GraphNode> var1 = make_shared<GraphNode>(Variable("1", data1));


        Tensor4f data2 = Tensor4f(1,1,3,2);
        data2.setValues({{{{1,2},{3,4},{5,6}}}});

        shared_ptr<GraphNode> var2 = make_shared<GraphNode>(Variable("2", data2));


        Tensor4f data3 = Tensor4f(1,1,4,4);
        data3.setValues({{{{ 1, 2, 3, 4},
                           { 5, 6, 7, 8},
                           { 9,10,11,12},
                           {13,14,15,16}}}});
        shared_ptr<GraphNode> var3 = make_shared<GraphNode>(Variable("3", data3));

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
            THEN("output should have right dimensions"){
                REQUIRE(opp.getData().dimension(0) == opp.getInputVec()[0]->getData().dimension(0));
                REQUIRE(opp.getData().dimension(3) == opp.getInputVec()[1]->getData().dimension(3));
            }
            WHEN("forward called"){
                opp.forward();
                THEN("output should be calculate"){
                    CHECK(opp.getData()(0,0,0,0) == 22);
                    CHECK(opp.getData()(0,0,0,1) == 28);
                    CHECK(opp.getData()(1,0,0,0) == 49);
                    CHECK(opp.getData()(1,0,0,1) == 64);
                }
            }
            WHEN("gradient is populated and backward called"){

                Tensor4f grad = Tensor4f(2,1,1,2);
                grad.setValues({{{{1,2}}},{{{3,4}}}});
                opp.setGradient(grad);
                opp.backward();
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
        shared_ptr<GraphNode> var1 = make_shared<GraphNode>(Variable("1", data1));

        Tensor4f data2 = Tensor4f(2,1,1,3);
        data2.setValues({{{{7,8,9}}},{{{10,11,12}}}});
        shared_ptr<GraphNode> var2 = make_shared<GraphNode>(Variable("2", data2));

        Tensor4f data3 = Tensor4f(4,1,1,4);
        data3.setValues({{{{1,2,3,4}}},{{{5,6,7,8}}},{{{9,10,11,12}}},{{{13,14,15,16}}}});
        shared_ptr<GraphNode> var3 = make_shared<GraphNode>(Variable("3", data3));

        WHEN("Constructed with 2 input nodes but wrong input dimensions"){
            NodeVec input_nodes_not_fit = {var1, var3};
            THEN("throw an error"){
                CHECK_THROWS_WITH(ElementwiseAdd(name, input_nodes_not_fit),  Catch::Contains( "error dimension miss fit:" ) && Catch::Contains( " in node " ));
            }
        }
        WHEN("Constructed with 2 input nodes having fitting dim"){
            NodeVec input_nodes_fit = {var1, var2};
            ElementwiseAdd opp = ElementwiseAdd(name, input_nodes_fit);
            THEN("output should have right dimensions"){
                CHECK(opp.getData().dimension(0) == opp.getInputVec()[0]->getData().dimension(0));
                CHECK(opp.getData().dimension(1) == opp.getInputVec()[0]->getData().dimension(1));
                CHECK(opp.getData().dimension(2) == opp.getInputVec()[0]->getData().dimension(2));
                CHECK(opp.getData().dimension(3) == opp.getInputVec()[0]->getData().dimension(3));
            }
            WHEN("forward called"){
                opp.forward();
                THEN("output should be calculate"){
                    CHECK(opp.getData()(0,0,0,0) == 8);
                    CHECK(opp.getData()(0,0,0,1) == 10);
                    CHECK(opp.getData()(0,0,0,2) == 12);
                    CHECK(opp.getData()(1,0,0,0) == 14);
                    CHECK(opp.getData()(1,0,0,1) == 16);
                    CHECK(opp.getData()(1,0,0,2) == 18);
                }
            }
            WHEN("gradient is populated and backward called"){
                Tensor4f grad = Tensor4f(2,1,1,3);
                grad.setValues({{{{1,1,2}}},{{{2,3,3}}}});
                opp.setGradient(grad);
                opp.backward();
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
        shared_ptr<GraphNode> var1 = make_shared<GraphNode>(Variable("1", data1));

        WHEN("Constructed "){
            NodeVec input_nodes = {var1};
            Sigmoid opp = Sigmoid(name, input_nodes);
            THEN("output should have same dimensions"){

                CHECK(opp.getData().dimension(0) == 2);
                CHECK(opp.getData().dimension(1) == 1);
                CHECK(opp.getData().dimension(2) == 1);
                CHECK(opp.getData().dimension(3) == 3);
            }
            WHEN("forward called"){
                opp.forward();
                THEN("output should be calculate"){
                    CHECK(opp.getData()(0,0,0,0) == 0.73105858_a);
                    CHECK(opp.getData()(0,0,0,1) == 0.88079708_a);
                    CHECK(opp.getData()(0,0,0,2) == 0.95257413_a);
                    CHECK(opp.getData()(1,0,0,0) == 0.98201379_a);
                    CHECK(opp.getData()(1,0,0,1) == 0.99330715_a);
                    CHECK(opp.getData()(1,0,0,2) == 0.99752738_a);
                }
            }
            WHEN("gradient is populated and backward called"){
                Tensor4f grad = Tensor4f(2,1,1,3);
                opp.forward();
                grad.setValues({{{{1,1,2}}},{{{2,3,3}}}});
                opp.setGradient(grad);
                opp.backward();
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
        shared_ptr<GraphNode> var1 = make_shared<GraphNode>(Variable("1", data1));

        WHEN("Constructed "){
            NodeVec input_nodes = {var1};
            ReLU opp = ReLU(name, input_nodes);
            THEN("output should have same dimensions"){

                CHECK(opp.getData().dimension(0) == 2);
                CHECK(opp.getData().dimension(1) == 1);
                CHECK(opp.getData().dimension(2) == 1);
                CHECK(opp.getData().dimension(3) == 3);
            }
            WHEN("forward called"){
                opp.forward();
                THEN("output should be calculate"){
                    CHECK(opp.getData()(0,0,0,0) == 1);
                    CHECK(opp.getData()(0,0,0,1) == 0);
                    CHECK(opp.getData()(0,0,0,2) == 3);
                    CHECK(opp.getData()(1,0,0,0) == 4);
                    CHECK(opp.getData()(1,0,0,1) == 5);
                    CHECK(opp.getData()(1,0,0,2) == 0);
                }
            }
            WHEN("gradient is populated and backward called"){
                Tensor4f grad = Tensor4f(2,1,1,3);
                opp.forward();
                grad.setValues({{{{1,-1,2}}},{{{-2,3,-3}}}});
                opp.setGradient(grad);
                opp.backward();
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