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
#include "libdl/Variable.h"
#include "libdl/opperation.h"
class Optimizer{
protected:
    vector<shared_ptr<Variable> > variable_vec;
public:
    explicit Optimizer(const vector<shared_ptr<Variable>> &variable_vec){Optimizer::variable_vec = variable_vec;}
    virtual void optimize()=0;
};

class SGD_Optimizer : Optimizer{
private:
    float learning_rate;
public:
    SGD_Optimizer(const vector<shared_ptr<Variable>> &variable_vec, float learning_rate):Optimizer(variable_vec){SGD_Optimizer::learning_rate=learning_rate;}
    void optimize() override;

};

#endif //TEST_OPTIMIZER_H
