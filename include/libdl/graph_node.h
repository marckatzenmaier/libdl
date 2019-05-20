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
using namespace std;
using namespace Eigen;

typedef struct Tensor{
    Eigen::MatrixXf data;
    Eigen::MatrixXf gradient;
}Tensor;
typedef shared_ptr<Tensor> TensorPtr;

class GraphNode{
    static int node_count;
protected:
    string name;
    MatrixXf data;
    MatrixXf gradient;
public:
    void setData(const MatrixXf &data);
    const MatrixXf &getGradient() const;
    void setGradient(const Eigen::MatrixXf &gradient);
    const MatrixXf &getData() const;
    GraphNode(const string& name, const MatrixXf& data = MatrixXf());
    const string &getName() const;
    virtual void forward(){}
    virtual void backward(){}
    virtual string getType(){return "GraphNode";}
};
typedef vector<shared_ptr<GraphNode> > NodeVec;








#endif //TEST_GRAPH_NODE_H
