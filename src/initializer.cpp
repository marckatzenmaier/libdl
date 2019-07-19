//
// Created by marc on 17.05.19.
//

//MatrixXd::Random(3,3)
#include "libdl/initializer.h"
#include "libdl/graph_node.h"
#include "libdl/Variable.h"
#include <random>
#include <cmath>

using namespace std;
using namespace Eigen;

void init_random(const shared_ptr<GraphNode>& variable){//xavier
    Tensor4f random(variable->getData().dimension(0),variable->getData().dimension(1),variable->getData().dimension(2),variable->getData().dimension(3));
    random.setRandom<Eigen::internal::NormalRandomGenerator<float>>();
    random = random / sqrt((float)(random.size()/variable->getData().dimension(3)));//xavier
    variable->setData(random);
}

void init_weights_random(const vector<shared_ptr<Variable>> &variable_vec){
    for(auto& i:variable_vec){
        init_random(i);
    }
}