//
// Created by marc on 17.05.19.
//

#ifndef TEST_LOSS_H
#define TEST_LOSS_H
#include "Eigen/Core"
#include <vector>
#include <libdl/graph.h>
#include "libdl/graph_node.h"
#include "libdl/Variable.h"
#include "libdl/opperation.h"

float loss_MSE(const shared_ptr<GraphNode>& output, const shared_ptr<GraphNode>& label){
    float n = output->getData().rows() * output->getData().cols();
    output->setGradient((output->getData()-label->getData())*2.f/n);
    return (output->getData()-label->getData()).array().square().mean();
}

#endif //TEST_LOSS_H
