//
// Created by marc on 18.05.19.
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
//shuffle (foo.begin(), foo.end(), std::default_random_engine(seed));


using namespace std;
using namespace Eigen;

int main()
{
    const int input_neurons = 2;
    const int hidden_neurons = 10;
    const int output_neurons = 1;
    // make network
    shared_ptr<GraphNode> input = make_shared<Variable>(Variable("input", MatrixXf(1, input_neurons)));

    shared_ptr<GraphNode> weights1 = make_shared<Variable>(Variable("weights1", MatrixXf(input_neurons,hidden_neurons)));
    shared_ptr<GraphNode> bias1 = make_shared<Variable>(Variable("bias1", MatrixXf(1, hidden_neurons)));
    shared_ptr<GraphNode> weights2 = make_shared<Variable>(Variable("weights2", MatrixXf(hidden_neurons,output_neurons)));

    shared_ptr<GraphNode> opp1 = make_shared<MatrixMultiplication>(MatrixMultiplication("MatMul_1", NodeVec{input, weights1}));
    shared_ptr<GraphNode> opp2 = make_shared<ElementwiseAdd>(ElementwiseAdd("EleAdd_1", NodeVec{opp1, bias1}));
    shared_ptr<GraphNode> opp3 = make_shared<Sigmoid>(Sigmoid("Sigmoid_1", NodeVec{opp2}));
    shared_ptr<GraphNode> opp4 = make_shared<MatrixMultiplication>(MatrixMultiplication("MatMul_2", NodeVec{opp3, weights2}));
    shared_ptr<GraphNode> opp5 = make_shared<Sigmoid>(Sigmoid("Sigmoid_2", NodeVec{opp4}));

    Graph graph = Graph(opp5);
    init_weights_random(graph.getWeights());
    SGD_Optimizer optim = SGD_Optimizer(graph.getWeights(), 0.1);


    shared_ptr<GraphNode> label = make_shared<Variable>(Variable("label", MatrixXf(1, output_neurons)));

    //set the input values
    pair<MatrixXf, MatrixXf> pair0,pair1,pair2,pair3;
    MatrixXf input_sample = MatrixXf(1, input_neurons);
    MatrixXf output_sample = MatrixXf(1, output_neurons);

    /*input_sample << 0,0;
    output_sample << 0;
    pair0.first = input_sample;
    pair0.second = output_sample;

    input_sample << 0,1;
    output_sample << 1;
    pair1.first = input_sample;
    pair1.second = output_sample;

    input_sample << 1,0;
    output_sample << 1;
    pair2.first = input_sample;
    pair2.second = output_sample;

    input_sample << 1,1;
    output_sample << 0;
    pair3.first = input_sample;
    pair3.second = output_sample;*/

    input_sample << -1,-1;
    output_sample << 0;
    pair0.first = input_sample;
    pair0.second = output_sample;

    input_sample << -1,1;
    output_sample << 1;
    pair1.first = input_sample;
    pair1.second = output_sample;

    input_sample << 1,-1;
    output_sample << 1;
    pair2.first = input_sample;
    pair2.second = output_sample;

    input_sample << 1,1;
    output_sample << 0;
    pair3.first = input_sample;
    pair3.second = output_sample;


    vector<pair<MatrixXf, MatrixXf>> data_set = {pair0, pair1, pair2, pair3};
    shuffle (data_set.begin(), data_set.end(), std::default_random_engine(0));

    for(int i = 0; i<10000; i++){
        float loss = 0.0;
        for(auto& d:data_set) {
            input->setData(d.first);
            label->setData(d.second);
            graph.forward();
            loss += loss_MSE(opp5, label);
            graph.backward();
            optim.optimize();
            //cout<<"Gradients"<<weights1->getGradient()<<"\n : "<< weights2->getGradient()<< "\n : "<<bias1->getGradient()<<endl<<"end gradients"<<endl;
            if(i==10000-1) {
                cout << d.first << " was feed into the net and it outputs: " << opp5->getData()
                     << " and it was supposed to output: " << d.second << endl;
            }
        }
        /*cout<<input->getName()<<endl<<"data \n"<<input->getData()<<endl<<"grad \n"<<input->getGradient()<<endl;
        cout<<weights1->getName()<<endl<<"data \n"<<weights1->getData()<<endl<<"grad \n"<<weights1->getGradient()<<endl;
        cout<<bias1->getName()<<endl<<"data \n"<<bias1->getData()<<endl<<"grad \n"<<bias1->getGradient()<<endl;
        cout<<opp1->getName()<<endl<<"data \n"<<opp1->getData()<<endl<<"grad \n"<<opp1->getGradient()<<endl;
        cout<<opp2->getName()<<endl<<"data \n"<<opp2->getData()<<endl<<"grad \n"<<opp2->getGradient()<<endl;
        cout<<opp3->getName()<<endl<<"data \n"<<opp3->getData()<<endl<<"grad \n"<<opp3->getGradient()<<endl;
        cout<<weights2->getName()<<endl<<"data \n"<<weights2->getData()<<endl<<"grad \n"<<weights2->getGradient()<<endl;
        cout<<opp4->getName()<<endl<<"data \n"<<opp4->getData()<<endl<<"grad \n"<<opp4->getGradient()<<endl;
        cout<<opp5->getName()<<endl<<"data \n"<<opp5->getData()<<endl<<"grad \n"<<opp5->getGradient()<<endl;*/
        //cout<<opp3->getData()<<endl;
        //cout<<weights1->getData()<<"\n : "<< weights2->getData()<< "\n : "<<bias1->getData()<<endl;
        shuffle (data_set.begin(), data_set.end(), std::default_random_engine(0));
        cout<<"loss in this epoch: "<< to_string(loss/data_set.size())<<endl;
        //cout<<endl<<endl;
    }
}