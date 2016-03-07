import sys
import os
import pickle
import numpy as np
from collections import defaultdict
from skimage import io
from skimage import transform
from skimage.transform import pyramid_gaussian

class Text_classification:
	
	def __init__(self):
		# print "Loading the CNN"
		self.nn = pickle.load(open('nn.pkl', 'rb')) 

	def put_image(self, image_path):
		# print "Loading the image"
		self.image = io.imread(image_path, as_grey=True)
		self.image = transform.resize(self.image,(50,50))
		self.image_scaled = io.imread(image_path, as_grey=True)
		self.image_scaled = transform.resize(self.image_scaled,(50,50))
		self.image_scaled *= (1/self.image_scaled.max())

	def load_image(self, img):
		self.image = img.copy()
		self.image = transform.resize(self.image,(50,50))
		self.image_scaled = self.image.copy()
		# self.image_scaled = transform.resize(self.image_scaled,(50,50))
		self.image_scaled *= (1/self.image_scaled.max())

	def generate_image_pyramids(self):
		# print "Generating image pyramids"
		rows, cols, dim = self.image.shape
		pyramid = tuple(pyramid_gaussian(self.image, downscale=2))
		return pyramid

	def sliding_window(self, image, window_size=[16,16]):
		pass

	def predict_class(self):
		return self.nn.predict(np.asarray([self.image_scaled]))