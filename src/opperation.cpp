//
// Created by marc on 11.05.19.
//

#include <Eigen/StdVector>
#include <iostream>
#include <iomanip>
#include <libdl/graph_node.h>
#include "libdl/opperation.h"

Opperation::Opperation(const std::__cxx11::basic_string<char> &name,
                       const vector<shared_ptr<GraphNode> > &inputs) : GraphNode(name, MatrixXf()) {
    Opperation::input_vec = inputs;
}

void Opperation::checkForNumInputs(int num_inputs){
    if(input_vec.size() < num_inputs){
        throw std::runtime_error("error to few ("+to_string(input_vec.size())+" expected "+to_string(num_inputs)+") input nodes in node: " + name);
    }
    else if(input_vec.size() > num_inputs){
        throw std::runtime_error("error to many ("+to_string(input_vec.size())+" expected "+to_string(num_inputs)+") input nodes in node: " + name);
    }
}

const NodeVec &Opperation::getInputVec() const {
    return input_vec;
}


MatrixMultiplication::MatrixMultiplication(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(2);
    if(input_vec[0]->getData().cols() != input_vec[1]->getData().rows()){
        throw std::runtime_error("error dimension miss fit: " + to_string(input_vec[0]->getData().cols())
        + " != " +to_string(input_vec[1]->getData().rows()) +" in node " + name);
    }
    this->setData(MatrixXf(input_vec[0]->getData().rows(), input_vec[1]->getData().cols()));
    this->setGradient(MatrixXf(input_vec[0]->getData().rows(), input_vec[1]->getData().cols())*0);
}

void MatrixMultiplication::forward(){
    this->setData(input_vec[0]->getData() * input_vec[1]->getData());
}

void MatrixMultiplication::backward() {
    input_vec[0]->setGradient(gradient*input_vec[1]->getData().transpose());
    input_vec[1]->setGradient(input_vec[0]->getData().transpose()*gradient);
}

ElementwiseAdd::ElementwiseAdd(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(2);
    if(input_vec[0]->getData().cols() != input_vec[1]->getData().cols()
        || input_vec[0]->getData().rows() != input_vec[1]->getData().rows()){
        throw std::runtime_error("error dimension miss fit: ("
        + to_string(input_vec[0]->getData().rows()) +"," + to_string(input_vec[0]->getData().cols())
        + ") != (" +to_string(input_vec[1]->getData().rows()) +"," + to_string(input_vec[1]->getData().cols())
        +") in node " + name);
    }
    this->setData(MatrixXf(input_vec[0]->getData().rows(), input_vec[0]->getData().cols()));
    this->setGradient(MatrixXf(input_vec[0]->getData().rows(), input_vec[0]->getData().cols())*0);
}

void ElementwiseAdd::forward(){
    this->setData(input_vec[0]->getData() + input_vec[1]->getData());
}

void ElementwiseAdd::backward() {
    input_vec[0]->setGradient(gradient);
    input_vec[1]->setGradient(gradient);
}

Sigmoid::Sigmoid(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(1);
    this->setData(MatrixXf(input_vec[0]->getData().rows(), input_vec[0]->getData().cols()));
    this->setGradient(MatrixXf(input_vec[0]->getData().rows(), input_vec[0]->getData().cols())*0);
}

void Sigmoid::forward(){
    //sig(x) = 1/(1+exp(-x))
    this->setData(1./(1.+(-input_vec[0]->getData()).array().exp()));
}

void Sigmoid::backward() {
    //dx: sig(x)*(1-sig(x))
    input_vec[0]->setGradient((data.array()*(1-data.array())).array() *gradient.array());
}

ReLU::ReLU(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(1);
    this->setData(MatrixXf(input_vec[0]->getData().rows(), input_vec[0]->getData().cols()));
    this->setGradient(MatrixXf(input_vec[0]->getData().rows(), input_vec[0]->getData().cols())*0);
}

void ReLU::forward(){
    //sig(x) = 1/(1+exp(-x))
    this->setData(input_vec[0]->getData().array().max(0.0));
}

void ReLU::backward(){
    input_vec[0]->setGradient((data.array()>0).cast<float>() * gradient.array());
    //input_vec[0]->setGradient((data.array()+1).cwiseMin(1).cwiseEqual(1).array().cast<float>() * gradient.array());
}