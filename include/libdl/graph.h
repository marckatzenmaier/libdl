//
// Created by marc on 11.05.19.
//

#ifndef TEST_GRAPH_H
#define TEST_GRAPH_H
#include "Eigen/Core"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "libdl/graph_node.h"
#include "libdl/opperation.h"
#include "libdl/placeholder.h"
#include "libdl/Constant.h"
#include "libdl/Variable.h"

typedef std::vector<Opperation> OpperationVec; //todo why i need this

class Graph{
private:
    void calc_forward_order();
    void calc_backward_order();
    std::vector<std::shared_ptr<GraphNode> > forward_order;
    std::vector<std::shared_ptr<GraphNode> > backward_order;
    std::shared_ptr<GraphNode> endpoint;
    std::vector<std::shared_ptr<Opperation> > opperation_vec;
    std::vector<std::shared_ptr<Variable> > variable_vec;
    std::vector<std::shared_ptr<Placeholder> > placeholder_vec;
    std::map<std::string, std::shared_ptr<Placeholder>> placeholder_map;
public:
    std::shared_ptr<GraphNode> getEndpoint(){return endpoint;}
    Graph(const std::shared_ptr<GraphNode>&  endpoint);
    Tensor4f forward();
    void backward();
    void setPlaceholder(std::vector<std::pair<std::string, Tensor4f>> feed_dict);
    std::vector<std::shared_ptr<Variable> > getWeights();
};

#endif //TEST_GRAPH_H
