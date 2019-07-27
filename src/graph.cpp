//
// Created by marc on 11.05.19.
//

#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <stack>
#include <queue>
#include <map>
#include <libdl/graph.h>

#include "libdl/graph_node.h"
#include "libdl/variable.h"
#include "libdl/opperation.h"

using namespace std;
using namespace Eigen;

struct search_graph_node{
    shared_ptr<GraphNode> node;
    bool visited = false;
    explicit search_graph_node(const shared_ptr<GraphNode>& node){search_graph_node::node=node;}
};

map<std::string, std::shared_ptr<Placeholder>> map_name_placeholder(std::vector<std::shared_ptr<Placeholder>> &placeholders){
    map<std::string, std::shared_ptr<Placeholder>> my_map;
    for(auto& v:placeholders){
        my_map.insert(std::pair<std::string, std::shared_ptr<Placeholder>>(v->getName(), v));
    }
    return my_map;
}

Graph::Graph(const shared_ptr<GraphNode>& endpoint) {
    Graph::endpoint = endpoint;
    calc_forward_order();
    calc_backward_order();
    placeholder_map = map_name_placeholder(placeholder_vec);

}

Tensor4f Graph::forward() {
    for(auto& i:forward_order){
        i->forward();
    }
    return endpoint->getData();
}

void Graph::backward() {
    for(auto& i:backward_order){
        i->backward();
    }

}

void Graph::clearGradients(){
    for(auto& i:variable_vec){
        i->clearGradient();
    }
}

void Graph::setPlaceholder(std::vector<std::pair<std::string, Tensor4f>> feed_dict) {
    for(auto & pair:feed_dict){
        placeholder_map[pair.first]->setData(pair.second);
    }

}

vector<shared_ptr<Variable> > Graph::getWeights() {
    return variable_vec;
}

void Graph::calc_forward_order() {
    std::stack<search_graph_node> node_stack;
    node_stack.push(search_graph_node(endpoint));
    string classType;

    while(!node_stack.empty()){
        if(!node_stack.top().visited){
            node_stack.top().visited = true;
            classType = node_stack.top().node->getType();
            if(classType=="Opperation"){
                std::shared_ptr<Opperation> node = std::static_pointer_cast<Opperation> (node_stack.top().node);
                NodeVec input_vec = node->getInputVec();
                for (auto it = input_vec.rbegin(); it != input_vec.rend(); ++it)
                {
                    node_stack.push(search_graph_node(*it));
                }
            }
            else if (classType=="Variable"){
                std::shared_ptr<Variable> node = std::static_pointer_cast<Variable> (node_stack.top().node);
                variable_vec.push_back(node);
                forward_order.push_back(node);
                node_stack.pop();
            }
            else if (classType=="Placeholder"){
                std::shared_ptr<Placeholder> node = std::static_pointer_cast<Placeholder> (node_stack.top().node);
                placeholder_vec.push_back(node);
                forward_order.push_back(node);
                node_stack.pop();
            }
        }
        else{
            forward_order.push_back(node_stack.top().node);
            node_stack.pop();
        }
    }
    /*cout<<"names of forward order"<<endl;
    for(auto& i:forward_order){
        cout<<i->getName()<<" ";
    }*/
    /*cout<<endl;
    cout<<"names of variables"<<endl;
    for(auto& i:variable_vec){
        cout<<i->getName()<<" ";
    }
    cout<<endl;*/
    //populate forward order with depth first search use a std::stack
    //and sort in the nodes in the right vector: variable_vec, placeholder_vec, opperation_vec

}

void Graph::calc_backward_order() {
    //populate backward order with breath first search use std::queue
    backward_order = vector<shared_ptr<GraphNode> >(forward_order.rbegin(), forward_order.rend());
    /*cout<<"names of backward order"<<endl;
    for(auto& i:backward_order){
        cout<<i->getName()<<" ";
    }
    cout<<endl;*/
}
