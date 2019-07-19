import numpy as np
import cv2
import my_dllib_py
import os
import glob
import random
path_prefix = '../extern/datasets/yalefaces_croped_32'
def contrastive_loss(first, second, label, margin=1.0):
	norm = np.linalg.norm(first.getData()-second.getData())
	loss_value = label * norm + (1-label) * max(0.0, margin**2 - norm)
	diff = first.getData()-second.getData()
	grad = label*2*diff + (1-label)*(0<margin**2 - norm) * -2*diff
	#print(diff)
	#print(grad)
	first.setGradient(grad)
	second.setGradient(-grad)
	return loss_value

def predict_same(first, second, margin=1.0):
	norm = np.linalg.norm(first.getData()-second.getData())
	return norm < margin**2


def subject_id(path):
	return int(os.path.basename(path)[7:9])

'''def getData(list_of_paths):
	paths1 = list_of_paths[0:len(list_of_paths)//2]
	paths2 = list_of_paths[len(list_of_paths)//2:len(list_of_paths)-len(list_of_paths) % 2]
	labels = [0]*len(paths1)
	for i in range(len(paths1)):
		labels[i] = int(subject_id(paths1[i]) == subject_id(paths2[i]))
	return paths1, paths2, labels'''

def getData(list_of_paths):
	identity = np.identity(len(list_of_paths)).reshape(-1)
	paths1 = np.tile(np.array(list_of_paths), (len(list_of_paths),1)).reshape(-1)[identity != 1.0]
	paths2 = np.tile(np.array(list_of_paths), (len(list_of_paths),1)).transpose().reshape(-1)[identity != 1.0]
	labels = [0]*len(paths1)
	for i in range(len(paths1)):
		labels[i] = int(subject_id(paths1[i]) == subject_id(paths2[i]))
	return paths1, paths2, labels

list_of_paths = glob.glob(path_prefix+'/*.png')
#random.shuffle(list_of_paths)
train = list_of_paths[0:121]
val = list_of_paths[121:165]
test = list_of_paths[145:165]
#train = ['1', '2', '3']
#p1, p2, lab = getData(train)
#for a,b,c in zip(p1,p2,lab):
#	print('{}\t{}\t{}'.format(a,b,c))
#exit()

leNet_simanese = my_dllib_py.make_LeNet_siamnese(1, 2)
my_dllib_py.init_weights_random(leNet_simanese[0].getWeights())
optim = my_dllib_py.SGD_Optimizer(leNet_simanese[0].getWeights(), 0.001)

random.shuffle(train)
#p1, p2, lab = getData(list_of_paths)
epochs=5
for e in range(epochs):

	#train part
	p1, p2, lab = getData(train)
	sum_equal_pair = np.array(lab).sum()
	#print(sum_equal_pair)
	chose_only_n_neg_pair = ((len(lab)-sum_equal_pair)//sum_equal_pair)
	epoch_len = len(p1)
	loss=0.0
	loss_t = 0.0
	t_ctr = 0
	loss_f = 0.0
	f_ctr = 0
	f_ctr_2 = 1
	ctr = 0
	for a,b,c in zip(p1, p2, lab):
		if c == 0 and f_ctr_2 < chose_only_n_neg_pair:
			f_ctr_2 += 1
			continue
		elif c == 0 and f_ctr_2 == chose_only_n_neg_pair:
			f_ctr_2 = 1
		img1 = np.expand_dims(np.expand_dims(cv2.imread(a, 0).astype(np.float32)/255.0, axis=0), axis=3)

		leNet_simanese[0].setPlaceholder([("input", img1)])
		leNet_simanese[0].forward()
		img2 = np.expand_dims(np.expand_dims(cv2.imread(b, 0).astype(np.float32)/255.0, axis=0), axis=3)
		leNet_simanese[1].setPlaceholder([("input", img2)])
		leNet_simanese[1].forward()
		leNet_simanese[0].clearGradients()
		leNet_simanese[1].clearGradients()

		l = contrastive_loss(leNet_simanese[0].getEndpoint(), leNet_simanese[1].getEndpoint(), c, 2.0)
		if c == 0:
			loss_f += l
			f_ctr += 1
		if c == 1:
			loss_t += l
			t_ctr += 1
		loss += l
		if l == 0.0:
			continue
		leNet_simanese[0].backward()
		leNet_simanese[1].backward()
		optim.optimize()
	if t_ctr == 0:
		t_ctr += 1
	if f_ctr == 0:
		f_ctr += 1
	print('loss train: {} \t\t: {} \t\t: {}'.format(loss/(f_ctr+t_ctr), loss_f/f_ctr, loss_t/t_ctr))
	random.shuffle(train)

	#validation part
	p1, p2, lab = getData(val)
	epoch_len = len(p1)
	loss = 0.0
	loss_t = 0.0
	t_ctr = 0
	loss_f = 0.0
	f_ctr = 0
	ctr = 0
	acc = 0
	acc_t = 0
	acc_f = 0
	for a, b, c in zip(p1, p2, lab):
		img1 = np.expand_dims(np.expand_dims(cv2.imread(a, 0).astype(np.float32) / 255.0, axis=0), axis=3)
		leNet_simanese[0].setPlaceholder([("input", img1)])
		leNet_simanese[0].forward()
		img2 = np.expand_dims(np.expand_dims(cv2.imread(b, 0).astype(np.float32) / 255.0, axis=0), axis=3)
		leNet_simanese[1].setPlaceholder([("input", img2)])
		leNet_simanese[1].forward()
		l = contrastive_loss(leNet_simanese[0].getEndpoint(), leNet_simanese[1].getEndpoint(), c, 2.0)
		ac = float(c == predict_same(leNet_simanese[0].getEndpoint(), leNet_simanese[1].getEndpoint(), 1.5))
		acc += ac
		if c == 0:
			loss_f += l
			f_ctr += 1
			acc_f += ac
		if c == 1:
			loss_t += l
			t_ctr += 1
			acc_t += ac
		loss += l
	if t_ctr == 0:
		t_ctr += 1
	if f_ctr == 0:
		f_ctr += 1
	print('loss val:{} \t\t: {} \t\t: {}'.format(loss / (f_ctr+t_ctr), loss_f / f_ctr, loss_t / t_ctr))
	print('acc: {} \t\t: {} \t\t: {}'.format(acc/(f_ctr+t_ctr), acc_f / f_ctr, acc_t / t_ctr))



my_dllib_py.save_weights("test.ckpt", leNet_simanese[0].getWeights())