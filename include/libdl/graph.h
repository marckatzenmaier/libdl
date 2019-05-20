//
// Created by marc on 11.05.19.
//

#ifndef TEST_GRAPH_H
#define TEST_GRAPH_H
#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include "libdl/graph_node.h"
#include "libdl/opperation.h"
#include "libdl/placeholder.h"
#include "libdl/Constant.h"
#include "libdl/Variable.h"

typedef vector<Opperation> OpperationVec;

class Graph{
private:
    void calc_forward_order();
    void calc_backward_order();
    vector<shared_ptr<GraphNode> > forward_order;
    vector<shared_ptr<GraphNode> > backward_order;
    shared_ptr<GraphNode> endpoint;
    vector<shared_ptr<Opperation> > opperation_vec;
    vector<shared_ptr<Variable> > variable_vec;
    vector<shared_ptr<Placeholder> > placeholder_vec;
public:
    Graph(shared_ptr<GraphNode>  endpoint);
    MatrixXf forward();
    void backward();
    void setPlaceholder();
    vector<shared_ptr<Variable> > getWeights();
};

#endif //TEST_GRAPH_H
