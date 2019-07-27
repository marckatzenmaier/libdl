//
// Created by marc on 11.06.19.
//

#include "libdl/helper_functions.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "libdl/variable.h"
#include "libdl/opperation.h"
#include "libdl/graph_node.h"
#include "libdl/graph.h"
#include "libdl/placeholder.h"


using namespace std;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<vector<float> > &vec)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            vector<float> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back((float)temp);
                }
            }
            vec.push_back(tp);
        }
    }
}

void read_Mnist_Label(string filename, vector<float> &vec)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec.push_back((float)temp);
        }
    }
}

void save_variable(std::string filename, std::shared_ptr<Variable> &variable){
    int name_length, data_length;
    string name = variable->getName();
    name_length = name.length();
    data_length = variable->getData().size();
    //cout << name_length << " " << data_length<< " " << name << " " << endl<<variable->getData()<<endl;
    std::ofstream ofile(filename, std::ios::binary);
    ofile.write((char*) &name_length, sizeof(int));
    ofile.write((char*) &data_length, sizeof(int));
    ofile.write((char*) name.c_str(), sizeof(char)*name_length);
    ofile.write((char*) variable->getData().data(), sizeof(float)*data_length);
    ofile.close();
}
void load_variable(std::string filename, std::shared_ptr<Variable> &variable){
    int name_length, data_length;
    ifstream ifile(filename, std::ios::binary);
    ifile.read((char*) &name_length, sizeof(int));
    ifile.read((char*) &data_length, sizeof(int));
    string name;
    name.resize(name_length);
    ifile.read((char*) name.c_str(), sizeof(char)*name_length);
    if(data_length != variable->getData().size()){
        cout<<"error wrong dimension loaded"<<endl;//todo make exception
    }
    ifile.read((char*) variable->getData().data(), sizeof(float)*data_length);
    //cout << name_length << " " << data_length<< " " << name << " " << endl<<variable->getData()<<endl;
    ifile.close();

}

map<std::string, std::shared_ptr<Variable>> map_name_variable(std::vector<std::shared_ptr<Variable>> &weights){
    map<std::string, std::shared_ptr<Variable>> my_map;
    for(auto& v:weights){
        my_map.insert(pair<std::string, std::shared_ptr<Variable>>(v->getName(), v));
    }
    return my_map;
}
void save_weights(std::string filename, std::vector<std::shared_ptr<Variable>> &weights){
    int name_length, data_length, num_of_variables;
    num_of_variables = weights.size();
    std::ofstream ofile(filename, std::ios::binary);
    ofile.write((char *) &num_of_variables, sizeof(int));
    for(auto& variable:weights) {
        string name = variable->getName();
        name_length = name.length();
        data_length = variable->getData().size();
        //cout << name_length << " " << data_length<< " " << name << " " << endl<<variable->getData()<<endl;
        ofile.write((char *) &name_length, sizeof(int));
        ofile.write((char *) &data_length, sizeof(int));
        ofile.write((char *) name.c_str(), sizeof(char) * name_length);
        ofile.write((char *) variable->getData().data(), sizeof(float) * data_length);
    }
    ofile.close();
}
void load_weights(std::string filename, std::vector<std::shared_ptr<Variable>> &weights){
    int name_length, data_length, num_of_variables;
    map<std::string, std::shared_ptr<Variable>> my_map = map_name_variable(weights);
    ifstream ifile(filename, std::ios::binary);
    ifile.read((char *) &num_of_variables, sizeof(int));
    for(int i = 0; i<num_of_variables;i++) {
        ifile.read((char *) &name_length, sizeof(int));
        ifile.read((char *) &data_length, sizeof(int));
        string name;
        name.resize(name_length);
        ifile.read((char *) name.c_str(), sizeof(char) * name_length);
        if (data_length != my_map[name]->getData().size()) {
            cout << "error wrong dimension loaded" << endl;//todo make exception
        }
        ifile.read((char *) my_map[name]->getData().data(), sizeof(float) * data_length);
    }
    //cout << name_length << " " << data_length<< " " << name << " " << endl<<variable->getData()<<endl;
    ifile.close();
}

Tensor4f copy_mnist_to_tensor(std::vector<std::vector<float> > imgs){// todo since only once not performance important remove memcpy
    int num_sample = imgs.size();
    int sample_size = imgs[0].size();
    Eigen::Tensor<float, 1, Eigen::RowMajor>data(num_sample * sample_size);
    for(int i = 0; i<num_sample;i++){
        memcpy(data.data()+i*sample_size, imgs[i].data(), sizeof(float)*sample_size);
    }
    Eigen::DSizes<Eigen::internal::traits<Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index, 4> reshape;
    reshape[0] = num_sample;
    reshape[1] = 28;
    reshape[2] = 28;
    reshape[3] = 1;
    Tensor4f data1 = data.reshape(reshape);
    return data1;

}
Tensor4f copy_mnist_label_to_tensor(std::vector<float> labels){
    int num_sample = labels.size();
    Tensor4f data(num_sample,1,1,10);
    data.setZero();
    for(int i= 0; i<num_sample;i++){
        data(i,0,0,(int)labels[i])=1;
    }
    return data;
}


MnistDataset::MnistDataset(const std::string &filepath_data, const std::string &filepath_label, int batch, int max_size) : Dataset(){
    read_Mnist(filepath_data, data_vec);
    read_Mnist_Label(filepath_label,label_vec);
    if(max_size <= 0) {
        index_vec.resize(data_vec.size());
    }
    else{
        index_vec.resize(max_size);
    }
    for(int i = 0; i<index_vec.size();i++){index_vec[i]=i;}
    this->batch = batch;
}
pair<Tensor4f, Tensor4f> MnistDataset::operator[](int index){
    Eigen::array<pair<int, int>, 4> paddings;//todo can be moved to constructor
    paddings[0] = make_pair(0, 0);
    paddings[1] = make_pair(2, 2);
    paddings[2] = make_pair(2, 2);
    paddings[3] = make_pair(0, 0);
    vector<vector<float>> batched_data;
    vector<float> batched_label;
    for(int i = 0;i<batch;i++){
        batched_data.push_back(data_vec[index_vec[index*batch+i]]);
        batched_label.push_back(label_vec[index_vec[index*batch+i]]);
    }
    Tensor4f devide(batch,32,32,1);
    devide.setConstant(255);
    Tensor4f temp = ( copy_mnist_to_tensor(batched_data).pad(paddings) / devide );
    pair<Tensor4f, Tensor4f> pair = make_pair(temp, copy_mnist_label_to_tensor(batched_label));
    return pair;
}

Graph make_LeNet(int b){
    shared_ptr<GraphNode> input = make_shared<Placeholder>(Placeholder("input", Tensor4f(b,32,32, 1)));
    shared_ptr<GraphNode> conv1_weights = make_shared<Variable>(Variable("conv1_weights", Tensor4f(5,5,1,6)));
    shared_ptr<GraphNode> conv1 = make_shared<Conv2d>(Conv2d("conv1", NodeVec{input, conv1_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv1_act = make_shared<TanH>(TanH("conv1_act", NodeVec{conv1}));

    shared_ptr<GraphNode> pool_average1 = make_shared<Pool_average>(Pool_average("pool_average1", NodeVec{conv1_act}));
    shared_ptr<GraphNode> pool_average1_act = make_shared<TanH>(TanH("pool_average1_act", NodeVec{pool_average1}));

    shared_ptr<GraphNode> conv2_weights = make_shared<Variable>(Variable("conv2_weights", Tensor4f(5,5,6,16)));
    shared_ptr<GraphNode> conv2 = make_shared<Conv2d>(Conv2d("conv2", NodeVec{pool_average1_act, conv2_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv2_act = make_shared<TanH>(TanH("conv2_act", NodeVec{conv2}));


    shared_ptr<GraphNode> pool_average2 = make_shared<Pool_average>(Pool_average("pool_average2", NodeVec{conv2_act}));
    shared_ptr<GraphNode> pool_average2_act = make_shared<TanH>(TanH("pool_average2_act", NodeVec{pool_average2}));

    shared_ptr<GraphNode> conv3_weights = make_shared<Variable>(Variable("conv3_weights", Tensor4f(5,5,16,120)));
    shared_ptr<GraphNode> conv3 = make_shared<Conv2d>(Conv2d("conv3", NodeVec{pool_average2_act, conv3_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv3_act = make_shared<TanH>(TanH("conv3_act", NodeVec{conv3}));

    shared_ptr<GraphNode> fc1_weights = make_shared<Variable>(Variable("fc1_weights", Tensor4f(1,1,120,84)));
    shared_ptr<GraphNode> fc1 = make_shared<MatrixMultiplication>(MatrixMultiplication("fc1", NodeVec{conv3_act, fc1_weights}));
    //todo add bias
    shared_ptr<GraphNode> fc1_act = make_shared<TanH>(TanH("fc1_act", NodeVec{fc1}));


    shared_ptr<GraphNode> fc2_weights = make_shared<Variable>(Variable("fc2_weights", Tensor4f(1,1,84,10)));
    shared_ptr<GraphNode> fc2 = make_shared<MatrixMultiplication>(MatrixMultiplication("fc2", NodeVec{fc1_act, fc2_weights}));
    //todo add bias
    shared_ptr<GraphNode> fc2_act = make_shared<Softmax>(Softmax("fc1_act", NodeVec{fc2}));
    return Graph(fc2_act);
}

Tensor4f argmax(Tensor4f &input){ // todo maybe rather math function
    int dim_0 = input.dimension(0)*input.dimension(1)*input.dimension(2);
    int dim_1 = input.dimension(3);
    //std::array<int,2> reshape = {dim_0, dim_1};
    Eigen::DSizes<Eigen::internal::traits<Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index, 2> reshape;
    reshape[0] = dim_0;
    reshape[1] = dim_1;
    Eigen::Tensor<float, 2, Eigen::RowMajor> reshaped_input = input.reshape(reshape);
    Eigen::Tensor<float, 1, Eigen::RowMajor> arg_max(dim_0);
    for(int i = 0; i<dim_0; i++){
        float max=reshaped_input(i,0);
        int idx = 0;
        for(int j = 1; j<dim_1; j++){
            if(max<reshaped_input(i,j)){
                max =reshaped_input(i,j);
                idx = j;
            }
        }
        arg_max(i)=idx;
    }

    Eigen::DSizes<Eigen::internal::traits<Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index, 4> reshape_out;
    reshape_out[0] = input.dimension(0);
    reshape_out[1] = input.dimension(1);
    reshape_out[2] = input.dimension(2);
    reshape_out[3] = 1;
    Tensor4f output = arg_max.reshape(reshape_out);
    return output;
}

float eval_accuracy(const Tensor4f &output, const Tensor4f &label){
    Tensor4f temp = (output == label).cast<float>();
    Eigen::Tensor<float, 0, Eigen::RowMajor> equals = temp.mean();
    return equals(0);
}

std::vector<Graph> make_LeNet_siamnese(int b, int output_dim){
    shared_ptr<GraphNode> input = make_shared<Placeholder>(Placeholder("input", Tensor4f(b,32,32, 1)));
    shared_ptr<GraphNode> conv1_weights = make_shared<Variable>(Variable("conv1_weights", Tensor4f(5,5,1,6)));
    shared_ptr<GraphNode> conv1 = make_shared<Conv2d>(Conv2d("conv1", NodeVec{input, conv1_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv1_act = make_shared<TanH>(TanH("conv1_act", NodeVec{conv1}));

    shared_ptr<GraphNode> pool_average1 = make_shared<Pool_average>(Pool_average("pool_average1", NodeVec{conv1_act}));
    shared_ptr<GraphNode> pool_average1_act = make_shared<TanH>(TanH("pool_average1_act", NodeVec{pool_average1}));

    shared_ptr<GraphNode> conv2_weights = make_shared<Variable>(Variable("conv2_weights", Tensor4f(5,5,6,16)));
    shared_ptr<GraphNode> conv2 = make_shared<Conv2d>(Conv2d("conv2", NodeVec{pool_average1_act, conv2_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv2_act = make_shared<TanH>(TanH("conv2_act", NodeVec{conv2}));


    shared_ptr<GraphNode> pool_average2 = make_shared<Pool_average>(Pool_average("pool_average2", NodeVec{conv2_act}));
    shared_ptr<GraphNode> pool_average2_act = make_shared<TanH>(TanH("pool_average2_act", NodeVec{pool_average2}));

    shared_ptr<GraphNode> conv3_weights = make_shared<Variable>(Variable("conv3_weights", Tensor4f(5,5,16,120)));
    shared_ptr<GraphNode> conv3 = make_shared<Conv2d>(Conv2d("conv3", NodeVec{pool_average2_act, conv3_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv3_act = make_shared<TanH>(TanH("conv3_act", NodeVec{conv3}));

    shared_ptr<GraphNode> fc1_weights = make_shared<Variable>(Variable("fc1_weights", Tensor4f(1,1,120,84)));
    shared_ptr<GraphNode> fc1 = make_shared<MatrixMultiplication>(MatrixMultiplication("fc1", NodeVec{conv3_act, fc1_weights}));
    //todo add bias
    shared_ptr<GraphNode> fc1_act = make_shared<TanH>(TanH("fc1_act", NodeVec{fc1}));


    shared_ptr<GraphNode> fc2_weights = make_shared<Variable>(Variable("fc2_weights", Tensor4f(1,1,84,output_dim)));
    shared_ptr<GraphNode> fc2 = make_shared<MatrixMultiplication>(MatrixMultiplication("fc2", NodeVec{fc1_act, fc2_weights}));
    //todo add bias
    //shared_ptr<GraphNode> fc2_act = make_shared<Softmax>(Softmax("fc1_act", NodeVec{fc2}));
    std::vector<Graph> output;
    output.push_back(Graph(fc2));





    shared_ptr<GraphNode> input_graph2 = make_shared<Placeholder>(Placeholder("input", Tensor4f(b,32,32, 1)));
    shared_ptr<GraphNode> conv1_graph2 = make_shared<Conv2d>(Conv2d("conv1", NodeVec{input_graph2, conv1_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv1_act_graph2 = make_shared<TanH>(TanH("conv1_act", NodeVec{conv1_graph2}));

    shared_ptr<GraphNode> pool_average1_graph2 = make_shared<Pool_average>(Pool_average("pool_average1", NodeVec{conv1_act_graph2}));
    shared_ptr<GraphNode> pool_average1_act_graph2 = make_shared<TanH>(TanH("pool_average1_act", NodeVec{pool_average1_graph2}));

    shared_ptr<GraphNode> conv2_graph2 = make_shared<Conv2d>(Conv2d("conv2", NodeVec{pool_average1_act_graph2, conv2_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv2_act_graph2 = make_shared<TanH>(TanH("conv2_act", NodeVec{conv2_graph2}));


    shared_ptr<GraphNode> pool_average2_graph2 = make_shared<Pool_average>(Pool_average("pool_average2", NodeVec{conv2_act_graph2}));
    shared_ptr<GraphNode> pool_average2_act_graph2 = make_shared<TanH>(TanH("pool_average2_act", NodeVec{pool_average2_graph2}));

    shared_ptr<GraphNode> conv3_graph2 = make_shared<Conv2d>(Conv2d("conv3", NodeVec{pool_average2_act_graph2, conv3_weights}));
    //todo add bias
    shared_ptr<GraphNode> conv3_act_graph2 = make_shared<TanH>(TanH("conv3_act", NodeVec{conv3_graph2}));

    shared_ptr<GraphNode> fc1_graph2 = make_shared<MatrixMultiplication>(MatrixMultiplication("fc1", NodeVec{conv3_act_graph2, fc1_weights}));
    //todo add bias
    shared_ptr<GraphNode> fc1_act_graph2 = make_shared<TanH>(TanH("fc1_act", NodeVec{fc1_graph2}));


    shared_ptr<GraphNode> fc2_graph2 = make_shared<MatrixMultiplication>(MatrixMultiplication("fc2", NodeVec{fc1_act_graph2, fc2_weights}));
    //todo add bias
    //shared_ptr<GraphNode> fc2_act_graph2 = make_shared<Softmax>(Softmax("fc1_act", NodeVec{fc2_graph2}));

    output.push_back(Graph(fc2_graph2));

    return output;
}