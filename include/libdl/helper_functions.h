//
// Created by marc on 11.06.19.
//

#ifndef TEST_HELPER_FUNCTIONS_H
#define TEST_HELPER_FUNCTIONS_H

#include <map>
#include "libdl/Variable.h"
#include "libdl/graph.h"
#include <vector>
#include <utility>

void read_Mnist(std::string filename, std::vector<std::vector<float> > &vec);
void read_Mnist_Label(std::string filename, std::vector<float> &vec);
void save_variable(std::string filename, std::shared_ptr<Variable> &variable);
void load_variable(std::string filename, std::shared_ptr<Variable> &variable);
void save_weights(std::string filename, std::vector<std::shared_ptr<Variable>> &weights);
void load_weights(std::string filename, std::vector<std::shared_ptr<Variable>> &weights);
std::map<std::string, std::shared_ptr<Variable>> map_name_variable(std::vector<std::shared_ptr<Variable>> &weights);
Tensor4f copy_mnist_label_to_tensor(std::vector<float> labels);
Tensor4f copy_mnist_to_tensor(std::vector<std::vector<float> > imgs);
Graph make_LeNet(int b);
Tensor4f argmax(Tensor4f &input);
float eval_accuracy(const Tensor4f &output, const Tensor4f &label);
std::vector<Graph> make_LeNet_siamnese(int b, int output_dim);

template <typename data, typename label>
class Dataset{
public:
    void shuffle(){std::shuffle(index_vec.begin(), index_vec.end(), urgn);}
    virtual std::pair<Tensor4f, Tensor4f> operator[](int index)=0;
    int size(){return index_vec.size()/batch;}//automatic drop last batch
    Dataset(){
        std::random_device rng;
        urgn = std::mt19937(rng());
    }
protected:
    std::mt19937 urgn;
    std::vector<data> data_vec;
    std::vector<label> label_vec;
    std::vector<int> index_vec;
    int batch;
};

class MnistDataset : public Dataset<std::vector<float>,float>{
public:
    MnistDataset(const std::string& filepath_data, const std::string& filepath_label, int batch, int max_size=-1);
    std::pair<Tensor4f, Tensor4f> operator[](int index) override;
};

#endif //TEST_HELPER_FUNCTIONS_H
