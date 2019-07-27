# my_dllib

The deep learning library is an implementiation of graph computation. It enables the user to define a graph and run it afterwards. It is written in modern C++ and uses CMake as build tool and containes python bindings for the main functionalities. As third party libraries it uses *Eigen*, *pybind11*, *Catch2*
It has two main components **GraphNode** and **Graph** objects and some additional functionalities for special deep learing pruposes.
- **GraphNode** are an interface which is used to build the graph structure. Its derived classes are:
    - **Placeholder** which are used to feed data in the graph when it is assembled
    - **Variable** which are the internal memory of the graph and which can be easily optimized for e.g. deep learning purposes
    - **Opperation** which perform mathematical operations on its input which can be placeholder, variables or/and opperations.
- **Graph** is an object which is needed to run the graph designed with **GraphNode** it will internaly handle the order of computiation for the forward and the backward pass and contain some useful functions for feeding data into placeholders or getting only the variables to optimize them
- Additional functionalities contain:
    - Optimizer class
    - Loss functions
    - Dataset class
    - initializer functions
    - wrapper functions for easier graph construction
    - some preconstructed network architectures
    - some preinstalled datasets
# Installation
For installing the library an installation of cmake is neccesary.
The first step is to download the repository with:
```
git clone git@gitlab.lrz.de:MarcKatzenmaier/libdl.git
```
Then create the build directory and change into it:
```
mkdir build
cd build
```
Afterwards run the cmake comand with e.g. the build_type:
```
cmake .. -DCMAKE_BUILD_TYPE=Release
```
And to build the repository run:
```
make
```
# Python bindings
The python bindings for this library won't contain the full capabilites of the library they are desinged with fast prototyping in mind. Therefore they don't contain the capability to extend c++ classes in python. If you want to write custom opperations you have to do this in c++. For the other common usecases like dataloading, parameter optimization and loss functions it is possible to do this in python using numpy due to the converstion from four dimensional eigen tensors in four dimensional numpy arrays. The Face Recognition example contains code for such a custom loss function and data loading in python.
# Extend the library
If you want to extend the library it is essential to derive from the corresponding base classes (e.g. opperation, optimizer) and use their interface to guarantee the compatibility with the library. For fast development it is possible to extend parts of the library with python functions for e.g. the contrastive loss like it is done in the Face Recognition example.
# Examples
- **XOR**
The XOR problem is a commen toy example for deep learning with the truth table:

    |Input1 | Input2 | Output|
    |------|------|------|
    |0|0|0|
    |0|1|1|
    |1|0|1|
    |1|1|0|

    This problem is only solvable with at least one hidden layer since a nonlinear seperation plane is needed to seperate the labels 1 and 0.
    The code for this example project is in the xor_problem.cpp file. It can be run with the `./XOR` comand form within the build directory.
- **MNIST**
    MNIST is a famous handwritten digits recognition task which is mostly used due to ist simplisity it contains of 60000 training images and 10000 test images with a resolution of 28*28 pixels. The task is to classify each image in one of the 10 classes. It archieves usually performances well a 80% accuracy. A model which was trained on 30000 images archieved 98.57% accuracy.
    The MNIST example project uses a LeNet architecture for classification and is randomly initialized.
    The code of the mnist problem can be found in mnist_problem.cpp. It is compiled during the build process and can be run from within the build folder with `./MNIST`
    It will train the network on 1000 images and then test it on the 10000 test images. As a result the training and validation loss will be displayed without any gui interface. It is ment to demonstrate the capability of the library to converge a more complex network
- **FACE RECOGNITION**
    This project aims for a simple Face Recognition Project with images from the [yale-face-database](http://vision.ucsd.edu/content/yale-face-database) which was croped to be squared and then resized to 32*32. This was done to reduce the computational time which is neccesary to the network. To tackle this task a siamese network architecture based on the LeNet architecture which was used for the MNIST dataset was used. Additionally a contrastive loss was implemented in python. There are several scripts to perform this task which are all copied during the build process in the build folder.
    - one is for training and save the network with two embedding dimensions`train_faceRecognition.py`
    - another is for visualizing a random input pair and displaying the embedding vector and predict if they belong to the same person `run_pair.py`
    - a third is to visualize for every test and training image the embedding vector color coded by subjcts `run_all_pairs.py` this displays the training since clusters of the same subject are close together
