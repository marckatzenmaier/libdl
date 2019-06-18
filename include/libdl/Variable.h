//
// Created by marc on 11.05.19.
//

#ifndef TEST_VARIABLE_H
#define TEST_VARIABLE_H
#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include "libdl/graph_node.h"

class Variable : public GraphNode{
public:
    Variable(const std::string &name, const Tensor4f &data);

    std::string getType() override {return "Variable";}
    void forward() override;

    void backward() override;
//practical same Variable Const Placeholder / just can be fed different
};


#endif //TEST_VARIABLE_H
