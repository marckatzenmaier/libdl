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

/**
 * calculates the mean square error based on 2 graph_nodes
 * @param output the output of the network in which should be backpropagated
 * @param label the label for the MSE in which isn't backpropagated
 * @return the MSE loss value
 */
float loss_MSE(const std::shared_ptr<GraphNode>& output, const std::shared_ptr<GraphNode>& label);
/**
 * calculates the crossentropy error
 * @param output the output of the network in which should be backpropagated
 * @param label the label for the Crossentropy error in which isn't backpropagated
 * @return the crossentropy loss value
 */
float loss_Crossentropy(const std::shared_ptr<GraphNode>& output, const std::shared_ptr<GraphNode>& label);

#endif //TEST_LOSS_H
