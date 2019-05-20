//
// Created by marc on 07.05.19.
//

#include "catch2/catch.hpp"
#include "libdl/opperation_node.h"
/*SCENARIO("Test basic operation node behavour", "[opperation_node]"){
    GIVEN("Instanciation of Operation Node"){

        class concreteOpperation : public opperation_node<int>{
        public:
        void forward() override {
            this->output->setData(this->first->getData());
        }
        void backward() override {
            this->first->setGradient(this->output->getGradient());
        }
        };
        concreteOpperation test_opp = concreteOpperation();
        WHEN("test getter and setter"){
            graph_node<int> test_input = graph_node<int>();
            graph_node<int> test_output = graph_node<int>();
            test_opp.setOutput(test_output);
            test_opp.setFirst(test_input);
            THEN("output should be set with the object itself no copy"){
                //REQUIRE(test_opp.getFirst() == test_input);
                //REQUIRE(test_opp.getOutput() == test_output);
            }
        }

        test_opp = concreteOpperation();
        WHEN("run forward"){
            test_opp.setFirst(graph_node<int>(5));
            test_opp.forward();
            THEN("output data should be populated if input data available"){
                REQUIRE(5==test_opp.getOutput().getData());
            }
        }

        WHEN("run backward"){
            graph_node<int> outWithGradient = graph_node<int>();
            outWithGradient.setGradient(5);
            test_opp.setOutput(outWithGradient);
            test_opp.backward();
            THEN("input gradients calculated if output gradient available"){
                REQUIRE(5==test_opp.getFirst().getGradient());
            }
        }

    }
}
SCENARIO( "Test the basic porperties of the graph_node class", "[graph_node]"){
    GIVEN("a data node of basic type"){
        graph_node<int> dataNode = graph_node<int>();
        WHEN("data and gradient is set to 5"){
            dataNode.setData(5);
            dataNode.setGradient(5);
            THEN("data and gradient should be 5"){
                REQUIRE(dataNode.getData() == 5);
                REQUIRE(dataNode.getGradient()==5);
            }
        }
        WHEN("graph_node initialized with 5") {
            dataNode = graph_node<int>(5);
            THEN("data need to be 5"){
                REQUIRE(dataNode.getData()==5);
            }
        }


    }
}*/