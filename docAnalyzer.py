from skimage.viewer import ImageViewer
from skimage import io, draw
import numpy as np
import sys
from test_image import Text_classification

class DocAnalyzer:
	def __init__(self, img_path, step_size_inp=32):
		self.txt = Text_classification()
		self.doc_image = io.imread(img_path, as_grey=True)
		self.parsing = ""
		self.step_size = int(step_size_inp)

	def castValue(self, num):
		'''
		To cast the predicted class labe to category { [0-9] or [A-Z] or [a-z] or ' ' }
		'''
		if num>=0 and num<=9:
			return str(num)
		elif num>=10 and num<=35:
			return str(chr(ord('A') -10 + num ))
		elif num>=36 and num<=61:
			return str(chr(ord('a') -36 + num ))
		else:
			return " "

	def parseDoc(self):
		h   = self.doc_image.shape[0]
		w 	= self.doc_image.shape[1]
		
		for i in range(0, h, self.step_size):
			for j in range(0, w, self.step_size):
				to_test = self.doc_image[i:i + self.step_size, j:j + self.step_size]
				# to_test = self.doc_image[j:j + self.step_size, i:i + self.step_size]
				self.txt.load_image(to_test)
				ret = self.txt.predict_class()[0][0]
				self.parsing += self.castValue(ret)
			self.parsing += '\n'
		print self.parsing
