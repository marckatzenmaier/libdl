//
// Created by marc on 29.07.19.
//
#include "catch2/catch.hpp"
#include "libdl/helper_functions.h"
#include <iostream>
#include <fstream>
using namespace Eigen;
using namespace std;
using namespace Catch::literals;

SCENARIO( "Test Helper functions briefly", "[Helper]") {
    WHEN("mnist dataset is given") {
        string filenameEvalImgs = "../extern/datasets/mnist/t10k-images-idx3-ubyte";
        string filenameEvalLabels = "../extern/datasets/mnist/t10k-labels-idx1-ubyte";

        std::vector<std::vector<float>> data_vec;
        std::vector<float> label_vec;
        read_Mnist(filenameEvalImgs, data_vec);
        read_Mnist_Label(filenameEvalLabels,label_vec);
        THEN("output should be like expected from MNIST") {
            CHECK(data_vec[5][460]  ==79);
            CHECK(data_vec[5][240]  ==254);
            CHECK(data_vec[1000][680]  ==19);
            CHECK(data_vec[1000][380]  ==254);
            CHECK(data_vec[5432][380]  ==103);
            CHECK(data_vec[5432][440]  ==116);
            CHECK(data_vec[9999][580]  ==52);
            CHECK(data_vec[9999][520]  ==86);
            CHECK(label_vec[49]  ==4);
            CHECK(label_vec[6568]==9);
            CHECK(label_vec[305] ==0);
            CHECK(label_vec[2383]==6);

        }
    }
    WHEN("construct networks") {
        THEN("LeNet does not throw error") {
            REQUIRE_NOTHROW(make_LeNet(1));
        }
        THEN("LeNet_siamese does not throw error") {
            REQUIRE_NOTHROW(make_LeNet_siamnese(2,2));
        }
    }
    WHEN("argmax is called") {

        Tensor4f input(1,1,1, 4);
        input.setValues({{{{1,2,3,4}}}});
        THEN("index belonging to max element should be returned") {
            CHECK(argmax(input)(0,0,0,0)==3);
        }
    }

    WHEN("construct MNIST Dataset") {
        string filenameIMG = "../extern/datasets/mnist/train-images-idx3-ubyte";
        string filenameLabels = "../extern/datasets/mnist/train-labels-idx1-ubyte";
        //MnistDataset dataset = MnistDataset(filenameIMG,filenameLabels,5, 1000);
        THEN("no error thrown") {
            REQUIRE_NOTHROW(MnistDataset(filenameIMG,filenameLabels,5, 1000));
            REQUIRE_NOTHROW(MnistDataset(filenameIMG,filenameLabels,5, -1));
        }
    }
    WHEN("getting first MNIST Data lable pair") {
        string filenameIMG = "../extern/datasets/mnist/train-images-idx3-ubyte";
        string filenameLabels = "../extern/datasets/mnist/train-labels-idx1-ubyte";
        MnistDataset dataset = MnistDataset(filenameIMG,filenameLabels,1, 1000);
        THEN("both label and data should be in a 4 dimensional tensor") {
            auto data_label_pair = dataset[0];
            CHECK(data_label_pair.first.dimensions().size()==4);
            CHECK(data_label_pair.second.dimensions().size()==4);

        }
    }
    WHEN("evaluating accuracy") {
        Tensor4f data = Tensor4f(1,1,1,4);
        data.setValues({{{{1,2,5,6}}}});
        Tensor4f label = Tensor4f(1,1,1,4);
        label.setValues({{{{0,2,4,6}}}});
        THEN("accuracy should be calculated correctly") {
            CHECK(eval_accuracy(data,label)==0.5f);
        }
    }
    GIVEN("Vector of Variable to store and load"){

        Tensor4f data1 = Tensor4f(2,1,1,3);
        data1.setValues({{{{1,2,3}}},{{{4,5,6}}}});
        shared_ptr<Variable> var1 = make_shared<Variable>(Variable("1", data1));
        vector<shared_ptr<Variable>> vec({var1});
        string save_path = "./temp_test_weights.ckpt";
        WHEN("saved"){
            save_weights(save_path, vec);
            fstream fileStream;
            fileStream.open(save_path);
            THEN("file created and exists"){
                CHECK(!fileStream.fail());
            }
        }
        WHEN("saved then change and then reload saved variables"){
            save_weights(save_path, vec);

            data1.setValues({{{{1,1,1}}},{{{1,1,1}}}});
            load_weights(save_path, vec);
            THEN("variable should be the same as in the beginning"){
                CHECK(var1->getData()(0,0,0,0)==1.f);
                CHECK(var1->getData()(0,0,0,1)==2.f);
                CHECK(var1->getData()(0,0,0,2)==3.f);
                CHECK(var1->getData()(1,0,0,0)==4.f);
                CHECK(var1->getData()(1,0,0,1)==5.f);
                CHECK(var1->getData()(1,0,0,2)==6.f);
            }
        }
        fstream fileStream;
        fileStream.open(save_path);
        if(!fileStream.fail()){
            std::remove(save_path.c_str());
        }
    }
}