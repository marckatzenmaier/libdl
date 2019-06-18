//
// Created by marc on 11.06.19.
//

#include "libdl/helper_functions.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "libdl/Variable.h"

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

Tensor4f copy_mnist_to_tensor(std::vector<std::vector<float> > imgs){
    int num_sample = imgs.size();
    int sample_size = imgs[0].size();
    Eigen::Tensor<float, 1, Eigen::RowMajor>data(num_sample * sample_size);
    for(int i = 0; i<num_sample;i++){
        memcpy(data.data()+i*sample_size, imgs[i].data(), sizeof(float)*sample_size);
    }
    Tensor4f data1 = data.reshape(array<int,4>({num_sample,28,28,1}));
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
template <typename data, typename label>
class Dataset{
public:
    void shuffle(){std::random_shuffle(index_vec.begin(), index_vec.end());}
    virtual pair<Tensor4f, Tensor4f> &operator[](int index)=0;
    int size(){return data_vec.size()/batch;}//automatic drop last batch
protected:
    vector<data> data_vec;
    vector<label> label_vec;
    vector<int> index_vec;
    int batch;
};

/*class MnistDataset : public Dataset<vector<float>,float>{
    MnistDataset(const string& filepath_data, const string& filepath_label, int batch){
        read_Mnist(filepath_data, data_vec);
        read_Mnist_Label(filepath_label,label_vec);
        index_vec.resize(data_vec.size());
        for(int i = 0; i<data_vec.size();i++){index_vec[i]=i;}
        this->batch = batch;
    }
    pair<Tensor4f, Tensor4f> &operator[](int index) override {
       vector<vector<float>> batched_data;
       vector<float> batched_label;
       for(int i = 0;i<batch;i++){
           batched_data.push_back(data_vec[index_vec[index*batch+i]]);
           batched_label.push_back(label_vec[index_vec[index*batch+i]]);
       }
       auto pair = make_pair(copy_mnist_to_tensor(batched_data), copy_mnist_label_to_tensor(batched_label));
       return pair;
    }
};*/