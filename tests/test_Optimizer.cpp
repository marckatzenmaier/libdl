//
// Created by marc on 17.05.19.
//
#include "catch2/catch.hpp"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include "libdl/Optimizer.h"

using namespace std;
using namespace Eigen;
SCENARIO( "Test Optimizer", "[Node]"){
    GIVEN("Vector of Variable with gradient, optimizer with lr 0.1"){

        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,2,3}}},{{{4,5,6}}}});
        Tensor4f grad = Tensor4f(2,1,1,3);
        grad.setValues({{{{1,1,1}}},{{{1,1,1}}}});
        shared_ptr<Variable> var1 = make_shared<Variable>(Variable("1", data1));
        var1->setGradient(grad);
        vector<shared_ptr<Variable>> vec({var1});
        SGD_Optimizer optim(vec, 0.1);
        WHEN("optimize called"){
            optim.optimize();
            THEN("weights updated based on gradient and learning rate"){
                CHECK(var1->getData()(0,0,0,0)==0.9f);
                CHECK(var1->getData()(0,0,0,1)==1.9f);
                CHECK(var1->getData()(0,0,0,2)==2.9f);
                CHECK(var1->getData()(1,0,0,0)==3.9f);
                CHECK(var1->getData()(1,0,0,1)==4.9f);
                CHECK(var1->getData()(1,0,0,2)==5.9f);
            }
        }
    }
}