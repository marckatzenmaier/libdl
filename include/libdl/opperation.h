//
// Created by marc on 11.05.19.
//

#ifndef TEST_OPPERATION_H
#define TEST_OPPERATION_H
#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include "libdl/graph_node.h"
#include <iostream>

class Opperation : public GraphNode{
public:
    Opperation(const string &name, const NodeVec &inputs);
    const NodeVec &getInputVec() const;
    string getType() override {return "Opperation";}

protected:
    NodeVec input_vec;
    void checkForNumInputs(int num_inputs);
};

class MatrixMultiplication : public  Opperation{
public:
    MatrixMultiplication(const string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;

};
class ElementwiseAdd : public  Opperation{
public:
    ElementwiseAdd(const string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};

class Sigmoid : public  Opperation{
public:
    Sigmoid(const string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};
class ReLU : public  Opperation{
public:
    ReLU(const string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};

#endif //TEST_OPPERATION_H
