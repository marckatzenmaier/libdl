//
// Created by marc on 11.06.19.
//

#ifndef TEST_HELPER_FUNCTIONS_H
#define TEST_HELPER_FUNCTIONS_H

#include <map>
#include "libdl/Variable.h"

void read_Mnist(std::string filename, std::vector<std::vector<float> > &vec);
void read_Mnist_Label(std::string filename, std::vector<float> &vec);
/*std::string serialize_variable(std::shared_ptr<Variable> &variable);
void deserialize_variable(std::shared_ptr<Variable> &variable, std::string data);*/
void save_variable(std::string filename, std::shared_ptr<Variable> &variable);
void load_variable(std::string filename, std::shared_ptr<Variable> &variable);
void save_weights(std::string filename, std::vector<std::shared_ptr<Variable>> &weights);
void load_weights(std::string filename, std::vector<std::shared_ptr<Variable>> &weights);
std::map<std::string, std::shared_ptr<Variable>> map_name_variable(std::vector<std::shared_ptr<Variable>> &weights);
Tensor4f copy_mnist_label_to_tensor(std::vector<float> labels);
Tensor4f copy_mnist_to_tensor(std::vector<std::vector<float> > imgs);

#endif //TEST_HELPER_FUNCTIONS_H
