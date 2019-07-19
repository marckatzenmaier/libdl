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
#include "libdl/Variable.h"

//typedef std::vector<Opperation> OpperationVec; //todo why i need this

/**
 * \brief the actual graph class which need to be instanciated for performing calculations
 *
 * it calculates the order in which forward and backward pass should be performed, data can be feed in the graph
 * using placeholders
 */
class Graph{
private:
    void calc_forward_order();
    void calc_backward_order();
    /**
     * order in which the forward pass is called
     */
    std::vector<std::shared_ptr<GraphNode> > forward_order;
    /**
     * order in which the backward pass is called
     */
    std::vector<std::shared_ptr<GraphNode> > backward_order;
    /**
     * endpoint of the graph and therefore contains the result of the forward pass
     */
    std::shared_ptr<GraphNode> endpoint;
    std::vector<std::shared_ptr<Opperation> > opperation_vec;
    /**
     * contains all variables which are part of the graph
     */
    std::vector<std::shared_ptr<Variable> > variable_vec;
    /**
     * contains all palceholders of the graph which can be feed
     */
    std::vector<std::shared_ptr<Placeholder> > placeholder_vec;
    std::map<std::string, std::shared_ptr<Placeholder>> placeholder_map;
public:
    /**
     * @return the endpoint of the graph which also can be used to set the gradient of the network
     */
    std::shared_ptr<GraphNode> getEndpoint(){return endpoint;}
    Graph(const std::shared_ptr<GraphNode>&  endpoint);
    /**
     * calculates the forward path of the the graph
     * @return the data of the endpoint
     */
    Tensor4f forward();
    /**
     * calculates the gradients for all graph_nodes based on the gradient of the endpoint
     */
    void backward();
    /**
     * used to feed data in the graph based on the name of the placeholder
     * @param feed_dict a vector of pairs which contain the name of the placeholder and its new value
     */
    void setPlaceholder(std::vector<std::pair<std::string, Tensor4f>> feed_dict);
    /**
     * @return all variables which are used in this graph
     */
    std::vector<std::shared_ptr<Variable> > getWeights();
    /**
     * sets all gradients of the variables to zero
     */
    void clearGradients();
};

#endif //TEST_GRAPH_H
