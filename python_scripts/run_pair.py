import numpy as np
import cv2
import my_dllib_py
import os
import glob
import random
import matplotlib.pyplot as plt
path_prefix = '../extern/datasets/yalefaces_croped_32'


def contrastive_loss(first, second, label, margin=1.0):
	norm = np.linalg.norm(first.getData()-second.getData())
	loss_value = label * norm + (1-label) * max(0.0, margin**2 - norm)
	diff = first.getData()-second.getData()
	grad = label*2*diff + (1-label)*(0<margin**2 - norm) * -2*diff
	first.setGradient(grad)
	second.setGradient(-grad)
	return loss_value


def predict_same(first, second, margin=1.0):
	norm = np.linalg.norm(first.getData()-second.getData())
	return norm < margin**2


def subject_id(path):
	return int(os.path.basename(path)[7:9])


def getData(list_of_paths):
	identity = np.identity(len(list_of_paths)).reshape(-1)
	paths1 = np.tile(np.array(list_of_paths), (len(list_of_paths),1)).reshape(-1)[identity != 1.0]
	paths2 = np.tile(np.array(list_of_paths), (len(list_of_paths),1)).transpose().reshape(-1)[identity != 1.0]
	labels = [0]*len(paths1)
	for i in range(len(paths1)):
		labels[i] = int(subject_id(paths1[i]) == subject_id(paths2[i]))
	return paths1, paths2, labels


list_of_paths = sorted(glob.glob(path_prefix+'/*.png'))
train = list_of_paths[0:121]
val = list_of_paths[121:165]
test = list_of_paths[145:165]

leNet_simanese = my_dllib_py.make_LeNet_siamnese(1, 2)
my_dllib_py.load_weights("test.ckpt", leNet_simanese[0].getWeights())

random.shuffle(train)
p1, p2, lab = getData(train)
a=p1[0]
b=p2[0]
c=lab[0]
img1 = np.expand_dims(np.expand_dims(cv2.imread(a, 0).astype(np.float32)/255.0, axis=0), axis=3)

leNet_simanese[0].setPlaceholder([("input", img1)])
leNet_simanese[0].forward()
img2 = np.expand_dims(np.expand_dims(cv2.imread(b, 0).astype(np.float32)/255.0, axis=0), axis=3)
leNet_simanese[1].setPlaceholder([("input", img2)])
leNet_simanese[1].forward()

l = contrastive_loss(leNet_simanese[0].getEndpoint(), leNet_simanese[1].getEndpoint(), c, 2.0)
print('they are the same: {} \nthe loss of this pair is {}, subjects ids are {} and {}'.format(predict_same(leNet_simanese[0].getEndpoint(), leNet_simanese[1].getEndpoint(),1.5), l, subject_id(a), subject_id(b)))
print('embeddings are first: {}  second: {}'.format(leNet_simanese[0].getEndpoint().getData(), leNet_simanese[1].getEndpoint().getData()))

plt.subplot(3,1,1)
plt.imshow(img1[0,:,:,0], cmap='gray')
plt.subplot(3,1,2)
plt.imshow(img2[0,:,:,0], cmap='gray')
plt.subplot(3,1,3)
plt.ylim(-15,15)
plt.xlim(-15,15)
plt.scatter(leNet_simanese[0].getEndpoint().getData()[0,0,0,0], leNet_simanese[0].getEndpoint().getData()[0,0,0,1])
plt.scatter(leNet_simanese[1].getEndpoint().getData()[0,0,0,0], leNet_simanese[1].getEndpoint().getData()[0,0,0,1])
plt.show()
