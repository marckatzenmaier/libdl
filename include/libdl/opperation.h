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

/**
 * \brief base class for all other opperations
 *
 * all other opperations derive from this class, it contains common functionality such as inputs
 */
class Opperation : public GraphNode{
public:
    Opperation(const std::string &name, const NodeVec &inputs);
    const NodeVec &getInputVec() const;
    std::string getType() override {return "Opperation";}

protected:
    NodeVec input_vec;
    void checkForNumInputs(int num_inputs);
};

/**
 * \brief opperation which performs the Matrix Multiplication
 *
 * in the common usecase of a fully connected layer the input is the first element of the input vector and the second is
 * the weights in form 1, 1, input_channels, output_channels
 */
class MatrixMultiplication : public  Opperation{
public:
    MatrixMultiplication(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;

};

/**
 * \brief opperation which performs the elementwise addition
 *
 * performes elementwise addition where both inputs need to have the same dimmension
 */
class ElementwiseAdd : public  Opperation{
public:
    ElementwiseAdd(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};

/**
 * \brief opperation which performs an elementwise sigmoid function
 */
class Sigmoid : public  Opperation{
public:
    Sigmoid(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};

/**
 * \brief opperation which performs elementwise ReLu: max(0,x)
 */
class ReLU : public  Opperation{
public:
    ReLU(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
private:
    Tensor4f zeros;
    Eigen::array<int, 4> bcast;
};

/**
 * \brief opperation which performs a 2d convolution
 *
 * in the common usecase of a convolutional layer the input is the first element of the input vector and the second is
 * the weights in form height, width, input_channels, output_channels
 */
class Conv2d : public  Opperation{
public:
    Conv2d(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;

};

/**
 * \brief opperation which performes average pooling
 */
class Pool_average : public  Opperation{
public:
    Pool_average(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};

/**
 * \brief opperation which performs the softmax function on the input
 */
class Softmax : public  Opperation{
public:
    Softmax(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};

/**
 * \brief opperation which performs an elementwise TanH function
 */
class TanH : public  Opperation{
public:
    TanH(const std::string &name, const NodeVec &inputs);
    void forward() override;
    void backward() override;
};
#endif //TEST_OPPERATION_H
