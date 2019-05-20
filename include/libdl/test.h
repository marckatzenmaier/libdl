//
// Created by marc on 26.04.19.
//

#ifndef TEST_TEST_H
#define TEST_TEST_H
#include <iostream>

#include "libdl/graph_node.h"
#include "libdl/Variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include <libdl/Optimizer.h>
#include "libdl/initializer.h"
#include "libdl/loss.h"


shared_ptr<GraphNode> make_Variable(const string& name, int rows, int cols){
    return make_shared<Variable>(Variable(name, MatrixXf(rows, cols)));
}

shared_ptr<GraphNode> make_MatrixMultiplication(const string& name, const shared_ptr<GraphNode>& mat_a, const shared_ptr<GraphNode>& mat_b){
    return make_shared<MatrixMultiplication>(MatrixMultiplication(name, NodeVec{mat_a, mat_b}));
}

shared_ptr<GraphNode> make_ElementwiseAdd(const string& name, const shared_ptr<GraphNode>& mat_a, const shared_ptr<GraphNode>& mat_b){
    return make_shared<ElementwiseAdd>(ElementwiseAdd(name, NodeVec{mat_a, mat_b}));
}

shared_ptr<GraphNode> make_Sigmoid(const string& name, const shared_ptr<GraphNode>& mat){
    return make_shared<Sigmoid>(Sigmoid(name, NodeVec{mat}));
}

shared_ptr<GraphNode> make_ReLU(const string& name, const shared_ptr<GraphNode>& mat){
    return make_shared<ReLU>(ReLU(name, NodeVec{mat}));
}

#endif //TEST_TEST_H
