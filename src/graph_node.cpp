//
// Created by marc on 06.05.19.
//

#include "libdl/graph_node.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>

using namespace std;
using namespace Eigen;

const std::string &GraphNode::getName() const {return name;}

void GraphNode::setData(const Tensor4f &data) {
    if((GraphNode::data.dimension(1)!=data.dimension(1) ||
            GraphNode::data.dimension(2)!=data.dimension(2) || GraphNode::data.dimension(3)!=data.dimension(3))
        && GraphNode::data.dimension(0)!=0 && GraphNode::data.dimension(1)!=0 && GraphNode::data.dimension(2)!=0
        && GraphNode::data.dimension(3)!=0){
        throw std::runtime_error("error data size changing forbidden in node " + name);
    }
    GraphNode::data = data;
}

const Tensor4f &GraphNode::getGradient() const {return gradient;}

void GraphNode::setGradient(const Tensor4f &gradient){
    if((gradient.dimension(1)!=data.dimension(1) ||
        gradient.dimension(2)!=data.dimension(2) || gradient.dimension(3)!=data.dimension(3))
       && GraphNode::gradient.dimension(0)!=0 && GraphNode::gradient.dimension(1)!=0
       && GraphNode::gradient.dimension(2)!=0 && GraphNode::gradient.dimension(3)!=0){
        throw std::runtime_error("error gradient size miss match in node" + name);
    }
    GraphNode::gradient = gradient;
}
void GraphNode::addGradient(const Tensor4f &gradient){
    if((gradient.dimension(0) != data.dimension(0) || gradient.dimension(1) != data.dimension(1) ||
        gradient.dimension(2) != data.dimension(2) || gradient.dimension(3) != data.dimension(3))
       && GraphNode::gradient.dimension(0)!=0 && GraphNode::gradient.dimension(1)!=0
       && GraphNode::gradient.dimension(2)!=0 && GraphNode::gradient.dimension(3)!=0){
        throw std::runtime_error("error gradient size miss match in node" + name);
    }
    GraphNode::gradient += gradient;
}
void GraphNode::clearGradient(){
    if (GraphNode::data.dimension(0) != GraphNode::gradient.dimension(0)){//todo not possible since batch change apears afterwards
        gradient = Tensor4f(data.dimension(0), data.dimension(1), data.dimension(2), data.dimension(3));
    }
    GraphNode::gradient.setZero();
}
const Tensor4f &GraphNode::getData() const {return data;}


GraphNode::GraphNode(const string& name){this->name = name;}
