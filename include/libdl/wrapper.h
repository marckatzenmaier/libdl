//
// Created by marc on 26.04.19.
//

#ifndef TEST_WRAPPER_H
#define TEST_WRAPPER_H
#include <iostream>

#include "libdl/graph_node.h"
#include "libdl/variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include <libdl/Optimizer.h>
#include "libdl/initializer.h"
#include "libdl/loss.h"

std::shared_ptr<GraphNode> make_Variable(const std::string& name, int rows, int cols){//todo fix this for tensor
    return std::make_shared<Variable>(Variable(name, Tensor4f(1, rows, cols, 1)));
}

std::shared_ptr<GraphNode> make_MatrixMultiplication(const std::string& name, const std::shared_ptr<GraphNode>& mat_a, const std::shared_ptr<GraphNode>& mat_b){
    return std::make_shared<MatrixMultiplication>(MatrixMultiplication(name, NodeVec{mat_a, mat_b}));
}

std::shared_ptr<GraphNode> make_ElementwiseAdd(const std::string& name, const std::shared_ptr<GraphNode>& mat_a, const std::shared_ptr<GraphNode>& mat_b){
    return std::make_shared<ElementwiseAdd>(ElementwiseAdd(name, NodeVec{mat_a, mat_b}));
}

std::shared_ptr<GraphNode> make_Sigmoid(const std::string& name, const std::shared_ptr<GraphNode>& mat){
    return std::make_shared<Sigmoid>(Sigmoid(name, NodeVec{mat}));
}

std::shared_ptr<GraphNode> make_ReLU(const std::string& name, const std::shared_ptr<GraphNode>& mat){
    return std::make_shared<ReLU>(ReLU(name, NodeVec{mat}));
}

#endif //TEST_WRAPPER_H
