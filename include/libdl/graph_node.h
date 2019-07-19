//
// Created by marc on 06.05.19.
//

#ifndef TEST_GRAPH_NODE_H
#define TEST_GRAPH_NODE_H

#include <vector>
#include <string>
#include <memory>
#include "unsupported/Eigen/CXX11/Tensor"

typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;

/**
 * \brief this is a base class for all nodes in the computational graph.
 *
 * Each node contain data(for the forward pass) a gradient(for the backward pass) and a name to be identified for
 * debugging.
 * Additionally to the get and set methods all base classes need to override the forward and the backward methods which
 * will be called in during the forward and backward pass
 * To identify the type a getType method is available which should be overritten so the graph can sort based on this
 * method
 *
 */
class GraphNode{
protected:
    std::string name;
    Tensor4f data;
    Tensor4f gradient;
public:
    void setData(const Tensor4f &data);
    const Tensor4f &getGradient() const;
    virtual void setGradient(const Tensor4f &gradient);
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
