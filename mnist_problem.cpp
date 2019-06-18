//
// Created by marc on 11.06.19.
//

#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/Variable.h"
#include "libdl/opperation.h"
#include <libdl/graph.h>
#include <libdl/Optimizer.h>
#include "libdl/initializer.h"
#include "libdl/loss.h"

#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include "unsupported/Eigen/CXX11/Tensor"
#include "libdl/helper_functions.h"

using namespace std;
/*Tensor4f copy_mnist_to_tensor(std::vector<std::vector<float> > imgs){
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
}*/
template <typename data, typename label>
class Dataset{
public:
    void shuffle(){std::random_shuffle(index_vec.begin(), index_vec.end());/* for(auto i:index_vec){cout<<i<<" ";}cout<<endl;*/}
    virtual pair<Tensor4f, Tensor4f> operator[](int index)=0;
    int size(){return data_vec.size()/batch;}//automatic drop last batch
protected:
    vector<data> data_vec;
    vector<label> label_vec;
    vector<int> index_vec;
    int batch;
};

class MnistDataset : public Dataset<vector<float>,float>{
public:
    MnistDataset(const string& filepath_data, const string& filepath_label, int batch){
        read_Mnist(filepath_data, data_vec);
        read_Mnist_Label(filepath_label,label_vec);
        index_vec.resize(1000/*data_vec.size()*/);
        for(int i = 0; i<1000/*data_vec.size()*/;i++){index_vec[i]=i;}//todo hack
        this->batch = batch;
    }
    pair<Tensor4f, Tensor4f> operator[](int index) override {
        Eigen::array<pair<int, int>, 4> paddings;
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
        //Tensor4f subtract(batch,32,32,1);
        //subtract.setConstant(0.5);
        Tensor4f temp = ( copy_mnist_to_tensor(batched_data).pad(paddings) / devide );
        pair<Tensor4f, Tensor4f> pair = make_pair(temp, copy_mnist_label_to_tensor(batched_label));
        return pair;
    }
};
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

int main(){

    string filenameIMG = "../extern/datasets/mnist/train-images-idx3-ubyte";
    string filenameLabels = "../extern/datasets/mnist/train-labels-idx1-ubyte";
    /*std::vector<std::vector<float> > imgs;
    std::vector<float> labels;
    read_Mnist(filenameIMG, imgs);
    Tensor4f inp = copy_mnist_to_tensor(imgs);
    read_Mnist_Label(filenameLabels,labels);
    Tensor4f lab = copy_mnist_label_to_tensor(labels);*/
    array<int, 4> input_offset = {0,0,0,0};
    array<int, 4> input_extend = {1,28,28,1};
    array<int, 4> label_offset = {0,0,0,0};
    array<int, 4> label_extend = {1,1,1,10};

    Eigen::array<pair<int, int>, 4> paddings;
    paddings[0] = make_pair(0, 0);
    paddings[1] = make_pair(2, 2);
    paddings[2] = make_pair(2, 2);
    paddings[3] = make_pair(0, 0);

    int batch = 10;
    MnistDataset dataset = MnistDataset(filenameIMG,filenameLabels,batch);
    Graph leNet = make_LeNet(batch);
    init_weights_random(leNet.getWeights());
    /*for(const auto& a : leNet.getWeights()){
        cout<<a->getData()<<endl;
    }*/
    SGD_Optimizer optim = SGD_Optimizer(leNet.getWeights(), 0.01);
    //Tensor4f input = inp.slice(input_offset, input_extend);//.eval().pad(paddings);//(batch,32,32,1);
    //Tensor4f input2 = input.pad(paddings);

    pair<Tensor4f, Tensor4f> sample = dataset[0];
    leNet.setPlaceholder(vector<std::pair<std::string, Tensor4f>>({make_pair("input", sample.first)}));
    //Tensor4f label_t = lab.slice(label_offset, label_extend);//(batch,1,1,10);
    shared_ptr<GraphNode> label = make_shared<Placeholder>(Placeholder("label", sample.second));
    //cout<<sample.first<<endl;
    label->setData(sample.second);
    cout<<sample.second<<endl;

    int batches =100;
    for(int i = 0; i<10; i++) {
        float loss=0.0;
        for(int a = 0; a<batches;a++) {
            sample = dataset[a];
            leNet.setPlaceholder(vector<std::pair<std::string, Tensor4f>>({make_pair("input", sample.first)}));
            label->setData(sample.second);

            leNet.forward();
            //cout<<leNet.getEndpoint()->getData()<<endl;
            loss += loss_Crossentropy(leNet.getEndpoint(), label);
            leNet.backward();

            //cout<<leNet.getEndpoint()->getGradient()<<endl;
            optim.optimize();
            //cout<<"label"<<sample.second<<endl;
            //cout<<"data "<<leNet.getEndpoint()->getData()<<endl;
        }
        //dataset.shuffle();
        cout<<loss/batches<<endl;
    }
    cout<<"label"<<sample.second<<endl;
    cout<<leNet.getEndpoint()->getData()<<endl;
}