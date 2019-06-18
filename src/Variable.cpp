//
// Created by marc on 11.05.19.
//

#include <libdl/graph_node.h>
#include "libdl/Variable.h"
using namespace std;
using namespace Eigen;

Variable::Variable(const std::string& name, const Tensor4f &data)
        : GraphNode(name) {
    setData(data);
}


void Variable::forward(){
}

void Variable::backward(){
}