//
// Created by marc on 17.05.19.
//

#ifndef TEST_OPTIMIZER_H
#define TEST_OPTIMIZER_H
#include "Eigen/Core"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <stack>
#include <queue>
#include <libdl/graph.h>

#include "libdl/graph_node.h"
#include "libdl/variable.h"
#include "libdl/opperation.h"

/**
 * \brief base class for all optimizers
 *
 * all derived classes need to override the optimize function
 */
class Optimizer{
protected:
    std::vector<std::shared_ptr<Variable> > variable_vec;
public:
    explicit Optimizer(const std::vector<std::shared_ptr<Variable>> &variable_vec){Optimizer::variable_vec = variable_vec;}
    virtual void optimize()=0;
};

/**
 * \brief Stochastic gradient descent optimizer
 *
 * optimizes the weight based on stochastic gradient descent algorithm
 */
class SGD_Optimizer : Optimizer{
private:
    float learning_rate;
public:
    SGD_Optimizer(const std::vector<std::shared_ptr<Variable>> &variable_vec, float learning_rate):Optimizer(variable_vec){SGD_Optimizer::learning_rate=learning_rate;}
    void optimize() override;

};

#endif //TEST_OPTIMIZER_H
