from sknn.mlp import Classifier, Convolution, Layer
from skimage import io
from skimage import transform
import os
import pickle
import numpy as np
import logging
logging.basicConfig()

path = '/home/ayush/Documents/CNN/English/Fnt/Sample0'

nn = Classifier(
    layers=[Convolution("Rectifier", channels=12, kernel_shape=(3,3)),
    		Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
    		Convolution("Rectifier", channels=4, kernel_shape=(3,3)),
    		Layer("Rectifier", units=64),
    		Layer("Softmax")],
    n_iter=10,
    learning_rate=0.002,
    verbose=True)
# n_iter=5,

X_train=list()
y_train=list()

print '-' * 50
print "LOADING THE DATA"
print '-' * 50

DATA_SET_VAL = 63

for i in range(DATA_SET_VAL):
	full_path=''
	if i+1<=9:
		full_path = path + '0' + str(i+1) + '/' 
	else:
		full_path = path + str(i+1) + '/'

	print full_path
	print "Added to the class" , i
	for image_name in os.listdir(full_path):
		image_path = full_path + image_name
		the_image = io.imread(image_path, as_grey=True)
		the_image = transform.resize(the_image,(50,50))
		the_image*=(1/the_image.max())
		X_train.append(the_image)
		y_train.append(i)

print '-' * 50
print "DATA LOADED"
print '-' * 50

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

nn.fit(X_train, y_train)

pickle.dump(nn, open('nn.pkl', 'wb'))