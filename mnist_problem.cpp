//
// Created by marc on 11.06.19.
//

#include "libdl/graph_node.h"
#include <iostream>
#include "libdl/variable.h"
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


            leNet.clearGradients();

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