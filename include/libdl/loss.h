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

float loss_MSE(const std::shared_ptr<GraphNode>& output, const std::shared_ptr<GraphNode>& label);
float loss_Crossentropy(const std::shared_ptr<GraphNode>& output, const std::shared_ptr<GraphNode>& label);

#endif //TEST_LOSS_H
