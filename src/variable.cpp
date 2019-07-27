//
// Created by marc on 11.05.19.
//

#include <libdl/graph_node.h>
#include "libdl/variable.h"
using namespace std;
using namespace Eigen;

Variable::Variable(const std::string& name, const Tensor4f &data)
        : GraphNode(name) {
    setData(data);
    gradient = Tensor4f(data.dimension(0),data.dimension(1),data.dimension(2),data.dimension(3));
    clearGradient();
}


void Variable::forward(){
}

void Variable::backward(){
}
void Variable::setGradient(const Tensor4f &gradient){
    addGradient(gradient);
};