//
// Created by marc on 17.05.19.
//

#ifndef TEST_INITIALIZER_H
#define TEST_INITIALIZER_H
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include <libdl/Optimizer.h>
#include "unsupported/Eigen/CXX11/Tensor"
void init_random(const std::shared_ptr<GraphNode>& variable);

/**
 * initializes all weigts random based on the xavier initialisation
 * @param variable_vec a vector of all variables which should be initialized
 */
void init_weights_random(const std::vector<std::shared_ptr<Variable>> &variable_vec);
#endif //TEST_INITIALIZER_H
