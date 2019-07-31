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

/**
 * \brief a class which is used for feeding information into a graph
 *
 * the name need to be unambigous within a graph
 */
class Placeholder : public GraphNode{
protected:
    void forward() override;
    void backward() override;
public:
    std::string getType() override;
    Placeholder(const std::string &name, const Tensor4f &data);
};

#endif //TEST_PLACEHOLDER_H
