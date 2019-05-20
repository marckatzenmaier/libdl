//
// Created by marc on 17.05.19.
//

#ifndef TEST_INITIALIZER_H
#define TEST_INITIALIZER_H
#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include <libdl/Optimizer.h>
void init_random(shared_ptr<GraphNode> variable);

void init_weights_random(const vector<shared_ptr<Variable>> &variable_vec);
#endif //TEST_INITIALIZER_H
