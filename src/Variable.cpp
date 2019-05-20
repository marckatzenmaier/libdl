//
// Created by marc on 11.05.19.
//

#include <libdl/graph_node.h>
#include "libdl/Variable.h"

Variable::Variable(const std::__cxx11::basic_string<char> &name, const Eigen::Matrix<float, Dynamic, Dynamic> &data)
        : GraphNode("Var_" + name, data) {}


void Variable::forward(){
}

void Variable::backward(){
}