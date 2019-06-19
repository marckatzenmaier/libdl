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
    int size(){return index_vec.size()/batch;}//automatic drop last batch
protected:
    vector<data> data_vec;
    vector<label> label_vec;
    vector<int> index_vec;
    int batch;
};

class MnistDataset : public Dataset<vector<float>,float>{
public:
    MnistDataset(const string& filepath_data, const string& filepath_label, int batch, int max_size=-1){
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
Tensor4f argmax(Tensor4f &input){
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
    //std::array<int,4> reshape_out = {(int)input.dimension(0),(int)input.dimension(1),(int)input.dimension(2),1};

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
int main(){

    string filenameIMG = "../extern/datasets/mnist/train-images-idx3-ubyte";
    string filenameLabels = "../extern/datasets/mnist/train-labels-idx1-ubyte";

    string filenameEvalImgs = "../extern/datasets/mnist/t10k-images-idx3-ubyte";
    string filenameEvalLabels = "../extern/datasets/mnist/t10k-labels-idx1-ubyte";
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
    MnistDataset dataset = MnistDataset(filenameIMG,filenameLabels,batch, 1000);
    MnistDataset eval_set =MnistDataset(filenameEvalImgs, filenameEvalLabels,batch);
    Graph leNet = make_LeNet(batch);
    init_weights_random(leNet.getWeights());
    SGD_Optimizer optim = SGD_Optimizer(leNet.getWeights(), 0.01);

    pair<Tensor4f, Tensor4f> sample;
    shared_ptr<GraphNode> label = make_shared<Placeholder>(Placeholder("label", sample.second));
    int epochs = 10;
    for(int i = 0; i<epochs; i++) {
        float loss=0.0;
        for(int a = 0; a<dataset.size();a++) {
            sample = dataset[a];
            leNet.setPlaceholder(vector<std::pair<std::string, Tensor4f>>({make_pair("input", sample.first)}));
            label->setData(sample.second);

            leNet.forward();
            loss += loss_Crossentropy(leNet.getEndpoint(), label);
            leNet.backward();

            optim.optimize();
        }
        dataset.shuffle();
        cout <<"train loss of epoch " << i<<" is "<<loss/dataset.size()<<endl;
        string savePath = "./lenet_"+to_string(i)+".ckpt";
        vector<shared_ptr<Variable> > weights = leNet.getWeights();
        save_weights(savePath, weights);
        float acc = 0, val_loss= 0;
        for(int a = 0; a<eval_set.size();a++){
            sample = eval_set[a];
            leNet.setPlaceholder(vector<std::pair<std::string, Tensor4f>>({make_pair("input", sample.first)}));

            leNet.forward();

            label->setData(sample.second);
            val_loss += loss_Crossentropy(leNet.getEndpoint(), label);

            Tensor4f output = leNet.getEndpoint()->getData();
            acc += eval_accuracy(argmax(output), argmax(sample.second));

        }
        cout <<"accuracy of epoch " << i<<" is "<<acc/eval_set.size()<<endl;
        cout <<"val loss of epoch " << i<<" is "<<val_loss/eval_set.size()<<endl<<endl;
    }
}