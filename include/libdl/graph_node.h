//
// Created by marc on 06.05.19.
//

#ifndef TEST_GRAPH_NODE_H
#define TEST_GRAPH_NODE_H

#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include "unsupported/Eigen/CXX11/Tensor"
//using namespace std;
//using namespace Eigen;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;

typedef struct Tensor{
    Tensor4f data;
    Tensor4f gradient;
    //Eigen::MatrixXf data;
    //Eigen::MatrixXf gradient;
}Tensor;
typedef std::shared_ptr<Tensor> TensorPtr;

class GraphNode{
    static int node_count;
protected:
    std::string name;
    Tensor4f data;
    Tensor4f gradient;
    //Eigen::MatrixXf data;
    //Eigen::MatrixXf gradient;
public:
    void setData(const Tensor4f &data);
    const Tensor4f &getGradient() const;
    void setGradient(const Tensor4f &gradient);
    void clearGradient();
    void addGradient(const Tensor4f &gradient);
    const Tensor4f &getData() const;
    GraphNode(const std::string& name);
    const std::string &getName() const;
    virtual void forward(){}
    virtual void backward(){}
    virtual std::string getType(){return "GraphNode";}
};
typedef std::vector<std::shared_ptr<GraphNode> > NodeVec;








#endif //TEST_GRAPH_NODE_H
