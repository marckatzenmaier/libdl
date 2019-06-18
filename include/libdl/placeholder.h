//
// Created by marc on 11.05.19.
//

#ifndef TEST_PLACEHOLDER_H
#define TEST_PLACEHOLDER_H
#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include "libdl/graph_node.h"

class Placeholder : public GraphNode{
public:
    void forward() override{};

    void backward() override{};

    std::string getType() override {return "Placeholder";}

    Placeholder(const std::string &name, const Tensor4f &data);
};

#endif //TEST_PLACEHOLDER_H
