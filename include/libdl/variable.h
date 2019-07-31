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

/**
 * \brief a class for variables of the graph
 *
 * this class is desinged to be optimized with the optimizer
 */
class Variable : public GraphNode{
protected:
    void forward() override;
    void backward() override;
public:
    Variable(const std::string &name, const Tensor4f &data);
    std::string getType() override;
    void setGradient(const Tensor4f &gradient) override;
};
#endif //TEST_VARIABLE_H
