//
// Created by marc on 11.05.19.
//

#ifndef TEST_CONSTANT_H
#define TEST_CONSTANT_H
#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include "libdl/graph_node.h"
class Constant : public GraphNode{
public:
    //Constant(const Tensor &innerState, const string &name = "");
    //Constant(const Eigen::MatrixXf &innerState, const string &name = "");
    //Constant(const float &innerState, const int &row, const int &col, const string &name = "");
    void forward() override;

    void backward() override;

private: Tensor innerState;
public:
    const Tensor &getInnerState() const;

private:
    static int const_count;
};
#endif //TEST_CONSTANT_H
