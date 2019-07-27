import numpy as np
import cv2
import my_dllib_py
import os
import glob
import matplotlib.pyplot as plt
path_prefix = '../extern/datasets/yalefaces_croped_32'

def subject_id(path):
    return int(os.path.basename(path)[7:9])


list_of_paths = sorted(glob.glob(path_prefix+'/*.png'))
train = list_of_paths[0:121]
val = list_of_paths[121:165]

colors = ['k','r','y','deepskyblue','b','m','orange','tan','grey','indigo','pink','teal','lawngreen','lime','g']

leNet_simanese = my_dllib_py.make_LeNet_siamnese(1, 2)
my_dllib_py.init_weights_random(leNet_simanese[0].getWeights())
optim = my_dllib_py.SGD_Optimizer(leNet_simanese[0].getWeights(), 0.001)

my_dllib_py.load_weights("test.ckpt", leNet_simanese[0].getWeights())

# calculate and plot all trainings samples
plt.figure()
plt.title('training')
plt.ylim(-15,15)
plt.xlim(-15,15)
data_points = np.zeros((len(train), 3))
end_point = leNet_simanese[0].getEndpoint()
for i, path in enumerate(train):
    img1 = np.expand_dims(np.expand_dims(cv2.imread(path, 0).astype(np.float32) / 255.0, axis=0), axis=3)
    leNet_simanese[0].setPlaceholder([("input", img1)])
    leNet_simanese[0].forward()
    data_points[i, 0:2] = end_point.getData()
    data_points[i, 2] = subject_id(path)
for i in range(data_points.shape[0]):
    plt.scatter(data_points[i, 0], data_points[i, 1], c=colors[int(data_points[i,2]-1)])
plt.show()


# calculate and plot all validation samples
plt.figure()
plt.title('validation')
plt.ylim(-15,15)
plt.xlim(-15,15)
data_points = np.zeros((len(val), 3))
end_point = leNet_simanese[0].getEndpoint()
for i, path in enumerate(val):
    img1 = np.expand_dims(np.expand_dims(cv2.imread(path, 0).astype(np.float32) / 255.0, axis=0), axis=3)
    leNet_simanese[0].setPlaceholder([("input", img1)])
    leNet_simanese[0].forward()
    data_points[i, 0:2] = end_point.getData()
    data_points[i, 2] = subject_id(path)
for i in range(data_points.shape[0]):
    plt.scatter(data_points[i, 0], data_points[i, 1], c=colors[int(data_points[i,2]-1)])
plt.show()


# calculate and plot all trainings and validation samples
plt.figure()
plt.title('training and validation')
plt.ylim(-15,15)
plt.xlim(-15,15)
data_points = np.zeros((len(list_of_paths), 3))
end_point = leNet_simanese[0].getEndpoint()
for i, path in enumerate(list_of_paths):
    img1 = np.expand_dims(np.expand_dims(cv2.imread(path, 0).astype(np.float32) / 255.0, axis=0), axis=3)
    leNet_simanese[0].setPlaceholder([("input", img1)])
    leNet_simanese[0].forward()
    data_points[i, 0:2] = end_point.getData()
    data_points[i, 2] = subject_id(path)
for i in range(data_points.shape[0]):
    plt.scatter(data_points[i, 0], data_points[i, 1], c=colors[int(data_points[i,2]-1)])
plt.show()