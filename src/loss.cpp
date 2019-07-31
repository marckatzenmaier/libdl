//
// Created by marc on 17.05.19.
//

#include "Eigen/Core"
#include <vector>
#include <libdl/graph.h>
#include "libdl/graph_node.h"
#include "libdl/variable.h"
#include "libdl/opperation.h"
float loss_MSE(const std::shared_ptr<GraphNode>& output, const std::shared_ptr<GraphNode>& label){
    float n = output->getData().size()/output->getData().dimension(0);
    output->setGradient((output->getData()-label->getData())*2.f/n);
    Eigen::Tensor<float, 0, Eigen::RowMajor> result = (output->getData()-label->getData()).square().mean();
    return result(0);
}
float loss_Crossentropy(const std::shared_ptr<GraphNode>& output, const std::shared_ptr<GraphNode>& label){
    output->setGradient((output->getData()-label->getData()));
    Tensor4f ones(output->getData().dimensions());
    ones.setConstant(1.0);
    Tensor4f epsilon(output->getData().dimensions());
    epsilon.setConstant(1e-7);
    Eigen::Tensor<float, 0, Eigen::RowMajor> result =
            -((label->getData()*(output->getData()+epsilon).log()) + ((ones-label->getData())*(ones-output->getData()+epsilon).log())).mean();
    return result(0);
}