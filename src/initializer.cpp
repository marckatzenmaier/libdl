//
// Created by marc on 17.05.19.
//

//MatrixXd::Random(3,3)
#include "libdl/initializer.h"
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include <libdl/Optimizer.h>
#include <random>

using namespace std;
using namespace Eigen;

void init_random(const shared_ptr<Variable>& variable){//xavier
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0,1.0);// /(variable->getData().rows()*variable->getData().cols()));
    auto normal = [&] (float) {return distribution(generator);};

    MatrixXf v = MatrixXf::NullaryExpr(variable->getData().rows(),variable->getData().cols(), normal );
    variable->setData(v);
    //variable->setData(MatrixXf::Random(variable->getData().rows(),variable->getData().cols()));
}//todo gaussian distribution

void init_weights_random(const vector<shared_ptr<Variable>> &variable_vec){
    for(auto& i:variable_vec){
        init_random(i);
    }
}