//
// Created by marc on 11.05.19.
//

#include <Eigen/StdVector>
#include <iostream>
#include <iomanip>
#include <libdl/graph_node.h>
#include "libdl/opperation.h"
#include "libdl/math_functions.h"

using namespace std;
using namespace Eigen;

Opperation::Opperation(const string &name,
                       const vector<shared_ptr<GraphNode> > &inputs) : GraphNode(name) {
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
    if(input_vec[0]->getData().dimension(3) != input_vec[1]->getData().dimension(2)){
        throw std::runtime_error("error dimension miss fit: " + to_string(input_vec[0]->getData().dimension(3))
        + " != " +to_string(input_vec[1]->getData().dimension(2)) +" in node " + name);
    }
    setData(Tensor4f(input_vec[0]->getData().dimension(0), 1, 1, input_vec[1]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0), 1, 1, input_vec[1]->getData().dimension(3)));
    clearGradient();
}

void MatrixMultiplication::forward(){
    setData(conv2d_NHWC(input_vec[0]->getData(), input_vec[1]->getData()));
}

void MatrixMultiplication::backward() {
    input_vec[0]->setGradient(conv2d_NHWC_backprop_input(gradient, input_vec[1]->getData()));
    input_vec[1]->setGradient(conv2d_NHWC_backprop_kernel(input_vec[0]->getData(), gradient));
}

ElementwiseAdd::ElementwiseAdd(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(2);
    if(input_vec[0]->getData().dimension(0) != input_vec[1]->getData().dimension(0) ||
            input_vec[0]->getData().dimension(1) != input_vec[1]->getData().dimension(1) ||
            input_vec[0]->getData().dimension(2) != input_vec[1]->getData().dimension(2) ||
            input_vec[0]->getData().dimension(3) != input_vec[1]->getData().dimension(3)
            ){
        throw std::runtime_error("error dimension miss fit: ("
        + to_string(input_vec[0]->getData().dimension(0)) +"," + to_string(input_vec[0]->getData().dimension(1))+","+
          to_string(input_vec[0]->getData().dimension(2)) +"," + to_string(input_vec[0]->getData().dimension(3))
        + ") != (" +to_string(input_vec[1]->getData().dimension(0)) +"," + to_string(input_vec[1]->getData().dimension(1))+","
                   +to_string(input_vec[1]->getData().dimension(2)) +"," + to_string(input_vec[1]->getData().dimension(3))
        +") in node " + name);
    }
    setData(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
            input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                               input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    clearGradient();
}

void ElementwiseAdd::forward(){
    setData(input_vec[0]->getData() + input_vec[1]->getData());
}

void ElementwiseAdd::backward() {
    input_vec[0]->setGradient(gradient);
    input_vec[1]->setGradient(gradient);
}

Sigmoid::Sigmoid(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(1);
    setData(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                           input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                               input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    clearGradient();
}

void Sigmoid::forward(){
    //sig(x) = 1/(1+exp(-x))
    setData(sigmoid(input_vec[0]->getData()));
}

void Sigmoid::backward() {
    //dx: sig(x)*(1-sig(x))
    input_vec[0]->setGradient((data*(1-data)) * gradient);
}

ReLU::ReLU(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(1);
    setData(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                     input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                         input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    clearGradient();
    zeros = Tensor4f(1,1,1,1);
    zeros.setZero();
    bcast = {(int)data.dimension(0), (int)data.dimension(1),(int)data.dimension(2),(int)data.dimension(3)};
}

void ReLU::forward(){
    setData((input_vec[0]->getData()>zeros.broadcast(bcast)).cast<float>()*input_vec[0]->getData());
}

void ReLU::backward(){
    input_vec[0]->setGradient((input_vec[0]->getData()>zeros.broadcast(bcast)).cast<float>() * gradient);
    //cout<<input_vec[0]->getName()<<input_vec[0]->getGradient()<<endl;
}

Conv2d::Conv2d(const std::string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(2);
    if(input_vec[0]->getData().dimension(3) != input_vec[1]->getData().dimension(2)){
        throw std::runtime_error("error dimension miss fit: " + to_string(input_vec[0]->getData().dimension(3))
                                 + " != " +to_string(input_vec[1]->getData().dimension(2)) +" in node " + name);
    }
    setData(Tensor4f(input_vec[0]->getData().dimension(0), input_vec[0]->getData().dimension(1)-input_vec[1]->getData().dimension(0)+1,
                     input_vec[0]->getData().dimension(2)-input_vec[1]->getData().dimension(1)+1, input_vec[1]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0), input_vec[0]->getData().dimension(1)-input_vec[1]->getData().dimension(0)+1,
                         input_vec[0]->getData().dimension(2)-input_vec[1]->getData().dimension(1)+1, input_vec[1]->getData().dimension(3)));
    clearGradient();
}

void Conv2d::forward(){
    setData(conv2d_NHWC(input_vec[0]->getData(), input_vec[1]->getData()));
}

void Conv2d::backward() {
    input_vec[0]->setGradient(conv2d_NHWC_backprop_input(gradient, input_vec[1]->getData()));
    input_vec[1]->setGradient(conv2d_NHWC_backprop_kernel(input_vec[0]->getData(), gradient));
}

Pool_average::Pool_average(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(1);
    int stride_h = 2, stride_w = 2;// todo only hacked like backward function
    setData(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1)/stride_h,
                     input_vec[0]->getData().dimension(2)/stride_w,input_vec[0]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1)/stride_h,
                         input_vec[0]->getData().dimension(2)/stride_w,input_vec[0]->getData().dimension(3)));
    clearGradient();
}

void Pool_average::forward(){
    setData(pool_average(input_vec[0]->getData(),2,2,2,2));
}

void Pool_average::backward(){
    input_vec[0]->setGradient(pool_average_backward(gradient,2,2,2,2));
}

Softmax::Softmax(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(1);
    setData(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                     input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                         input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    clearGradient();
}

void Softmax::forward(){
    setData(softmax(input_vec[0]->getData()));
}

void Softmax::backward() {
    input_vec[0]->setGradient(softmax_backward(data, gradient));
}

TanH::TanH(const string &name, const NodeVec &inputs) : Opperation(name, inputs) {
    checkForNumInputs(1);
    setData(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                     input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    setGradient(Tensor4f(input_vec[0]->getData().dimension(0),input_vec[0]->getData().dimension(1),
                         input_vec[0]->getData().dimension(2),input_vec[0]->getData().dimension(3)));
    clearGradient();
}

void TanH::forward(){
    Tensor4f two(data.dimensions());
    two.setConstant(2.0);
    Tensor4f one(data.dimensions());
    one.setConstant(1.0);
    Tensor4f in_two = input_vec[0]->getData() * two;
    Tensor4f sig_tow = (sigmoid(in_two)*two) - one;
    setData(sig_tow);
}

void TanH::backward() {

    Tensor4f one(data.dimensions());
    one.setConstant(1.0);
    input_vec[0]->setGradient((one -(data*data)) * gradient);
}