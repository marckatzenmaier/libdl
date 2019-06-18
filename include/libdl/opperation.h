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
    Opperation(const std::string &name, const NodeVec &inputs);
    const NodeVec &getInputVec() const;
    std::string getType() override {return "Opperation";}

protected:
    NodeVec input_vec;
    void checkForNumInputs(int num_inputs);
};

class MatrixMultiplication : public  Opperation{
public:
    MatrixMultiplication(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;

};
class ElementwiseAdd : public  Opperation{
public:
    ElementwiseAdd(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};

class Sigmoid : public  Opperation{
public:
    Sigmoid(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};
class ReLU : public  Opperation{
public:
    ReLU(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
private:
    Tensor4f zeros;
    Eigen::array<int, 4> bcast;
};
class Conv2d : public  Opperation{
public:
    Conv2d(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;

};
class Pool_average : public  Opperation{
public:
    Pool_average(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};
class Softmax : public  Opperation{
public:
    Softmax(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};
class TanH : public  Opperation{
public:
    TanH(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};
#endif //TEST_OPPERATION_H
