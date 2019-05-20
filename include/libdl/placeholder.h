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

    string getType() override {return "Placeholder";}
private:
    Tensor innerState;
};

#endif //TEST_PLACEHOLDER_H
