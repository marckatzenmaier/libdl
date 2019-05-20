//
// Created by marc on 06.05.19.
//

#include "libdl/graph_node.h"
#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include <iostream>

// graphnode starts here
int GraphNode::node_count = 0;

const std::string &GraphNode::getName() const {
    return name;
}

void GraphNode::setData(const MatrixXf &data) {
    if((GraphNode::data.rows()!=data.rows() || GraphNode::data.cols()!=data.cols())
        && GraphNode::data.rows()!=0 && GraphNode::data.cols()!=0){
        throw std::runtime_error("error data size changing forbidden in node " + name);
    }
    GraphNode::data = data;
}

const MatrixXf &GraphNode::getGradient() const {
    return gradient;
}

void GraphNode::setGradient(const Eigen::MatrixXf &gradient){
    if((gradient.rows()!=data.rows() || gradient.cols()!=data.cols())
       && GraphNode::gradient.rows()!=0 && GraphNode::gradient.cols()!=0){
        throw std::runtime_error("error gradient size miss match in node" + name);
    }
    GraphNode::gradient = gradient;
}
const Eigen::MatrixXf &GraphNode::getData() const {
    return data;
}


GraphNode::GraphNode(const string& name, const MatrixXf& data){
    this->name = name;
    this->data = data;
    this->gradient = MatrixXf(data.rows(), data.cols());

}

/*int Constant::const_count = 0;
void Constant::forward() {
    GraphNode::forward();
}

void Constant::backward() {
    GraphNode::backward();
}*/

/*Constant::Constant(const Tensor &innerState, const string &name){
    if(name == ""){
        this->name = "const_"+to_string(const_count);
        const_count++;
    }
    this->innerState = innerState;
}
Constant::Constant(const Eigen::MatrixXf &innerState, const string &name) {
    if(name == ""){
        this->name = "const_"+to_string(const_count);
        const_count++;
    }
    this->innerState.data = innerState;

}
Constant::Constant(const float &innerState, const int &row, const int &col, const string &name) {
    if(name == ""){
        this->name = "const_"+to_string(const_count);
        const_count++;
    }
    this->innerState.data = Eigen::MatrixXf(row,col).array() + innerState;

}*/

/*const Tensor &Constant::getInnerState() const {
    return innerState;
}

void Placeholder::forward() {
    GraphNode::forward();
}

void Placeholder::backward() {
    GraphNode::backward();
}

void Variable::forward() {
    GraphNode::forward();
}

void Variable::backward() {
    GraphNode::backward();
}

void Opperation::forward() {
    GraphNode::forward();
}

void Opperation::backward() {
    GraphNode::backward();
}
*/