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
//using namespace std;
//using namespace Eigen;

typedef struct Tensor{
    Eigen::MatrixXf data;
    Eigen::MatrixXf gradient;
}Tensor;
typedef std::shared_ptr<Tensor> TensorPtr;

class GraphNode{
    static int node_count;
protected:
    std::string name;
    Eigen::MatrixXf data;
    Eigen::MatrixXf gradient;
public:
    void setData(const Eigen::MatrixXf &data);
    const Eigen::MatrixXf &getGradient() const;
    void setGradient(const Eigen::MatrixXf &gradient);
    const Eigen::MatrixXf &getData() const;
    GraphNode(const std::string& name, const Eigen::MatrixXf& data = Eigen::MatrixXf());
    const std::string &getName() const;
    virtual void forward(){}
    virtual void backward(){}
    virtual std::string getType(){return "GraphNode";}
};
typedef std::vector<std::shared_ptr<GraphNode> > NodeVec;








#endif //TEST_GRAPH_NODE_H
