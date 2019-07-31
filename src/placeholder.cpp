//
// Created by marc on 11.05.19.
//

#include <libdl/graph_node.h>
#include "libdl/placeholder.h"
using namespace std;
using namespace Eigen;

std::string Placeholder::getType(){return "Placeholder";}
Placeholder::Placeholder(const std::string& name, const Tensor4f &data)
        : GraphNode(name) {
    setData(data);
}

void Placeholder::forward(){
}

void Placeholder::backward(){
}
