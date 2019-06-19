# libdl

To run the XOR problem: just make the project and run the ./XOR file which was build
it will produce exact the same output since the random seed for the initialization is set to 0

To run the MNIST problem(for the assingment it is included in the yml file):
just make the project and run the ./MNIST file which was build. It will train on the first 1000 images on the
mnist dataset and evaluate on the whole eval set. It archieves usually performances well a 80% accuracy. A model which
was trained on 30000 images archieved 98.57% accuracy. (the validation loss is in the beginning less than the train loss
since the train loss is only the average of the loss within the epoch and the validation loss uses the state after
training on one epoch.