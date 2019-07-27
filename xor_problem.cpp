//
// Created by marc on 18.05.19.
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

#include "libdl/math_functions.h"

#include <ctime>

using namespace std;
using namespace Eigen;
template <typename T>
using ConstEigenArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

void printTensorNHWC(const Eigen::Tensor<float, 4, RowMajor>& tensor){
    cout<<"[";
    for(int n = 0; n<tensor.dimensions()[0]; n++){
        cout<<"[";
        for(int h = 0; h<tensor.dimensions()[1]; h++){
            cout<<"[";
            for(int w = 0; w<tensor.dimensions()[2]; w++){
                cout<<"[ ";
                for(int c = 0; c<tensor.dimensions()[3]; c++){
                    cout<<tensor(n,h,w,c)<<" ";
                }
                cout<<"], ";
            }
            cout<<"]" << endl<<"  ";
        }
        cout<<"]"<<endl<<"  ";
    }
    cout<<endl<<"]"<<endl;
}

int main()
{
    const int input_neurons = 2;
    const int hidden_neurons = 10;
    const int output_neurons = 1;
    // make network
    shared_ptr<GraphNode> input = make_shared<Variable>(Variable("input", Tensor4f(1,1,1, input_neurons)));

    shared_ptr<GraphNode> weights1 = make_shared<Variable>(Variable("weights1", Tensor4f(1,1,input_neurons,hidden_neurons)));
    shared_ptr<GraphNode> bias1 = make_shared<Variable>(Variable("bias1", Tensor4f(1,1,1, hidden_neurons)));
    shared_ptr<GraphNode> weights2 = make_shared<Variable>(Variable("weights2", Tensor4f(1,1,hidden_neurons,output_neurons)));

    shared_ptr<GraphNode> opp1 = make_shared<MatrixMultiplication>(MatrixMultiplication("MatMul_1", NodeVec{input, weights1}));
    shared_ptr<GraphNode> opp2 = make_shared<ElementwiseAdd>(ElementwiseAdd("EleAdd_1", NodeVec{opp1, bias1}));
    shared_ptr<GraphNode> opp3 = make_shared<Sigmoid>(Sigmoid("Sigmoid_1", NodeVec{opp2}));
    shared_ptr<GraphNode> opp4 = make_shared<MatrixMultiplication>(MatrixMultiplication("MatMul_2", NodeVec{opp3, weights2}));
    shared_ptr<GraphNode> opp5 = make_shared<Sigmoid>(Sigmoid("Sigmoid_2", NodeVec{opp4}));

    Graph graph = Graph(opp5);
    init_weights_random(graph.getWeights());
    SGD_Optimizer optim = SGD_Optimizer(graph.getWeights(), 0.1);


    shared_ptr<GraphNode> label = make_shared<Variable>(Variable("label", Tensor4f(1, 1, 1, output_neurons)));

    //set the input values
    pair<Tensor4f, Tensor4f> pair0,pair1,pair2,pair3;
    Tensor4f input_sample = Tensor4f(1, 1,1, input_neurons);
    Tensor4f output_sample = Tensor4f(1,1,1, output_neurons);

    input_sample.setValues({{{{-1,-1}}}});
    output_sample.setValues({{{{0}}}});
    pair0.first = input_sample;
    pair0.second = output_sample;


    input_sample.setValues({{{{-1,1}}}});
    output_sample.setValues({{{{1}}}});
    pair1.first = input_sample;
    pair1.second = output_sample;


    input_sample.setValues({{{{1,-1}}}});
    output_sample.setValues({{{{1}}}});
    pair2.first = input_sample;
    pair2.second = output_sample;


    input_sample.setValues({{{{1,1}}}});
    output_sample.setValues({{{{0}}}});
    pair3.first = input_sample;
    pair3.second = output_sample;


    vector<pair<Tensor4f, Tensor4f>> data_set = {pair0, pair1, pair2, pair3};
    shuffle (data_set.begin(), data_set.end(), std::default_random_engine(0));

    for(int i = 0; i<10000 ; i++){
        float loss = 0.0;
        for(auto& d:data_set) {
            input->setData(d.first);
            label->setData(d.second);
            graph.clearGradients();
            graph.forward();
            loss += loss_MSE(opp5, label);
            graph.backward();
            optim.optimize();
            if(i==10000-1) {
                cout << d.first << " was feed into the net and it outputs: " << opp5->getData()
                     << " and it was supposed to output: " << d.second << endl;
            }
        }
        shuffle (data_set.begin(), data_set.end(), std::default_random_engine(0));
        cout<<"loss in this epoch: "<< to_string(loss/data_set.size())<<endl;
    }
}