from skimage.viewer import ImageViewer
from skimage import io, draw
import numpy as np
import sys
from test_image import Text_classification

txt = Text_classification()

def window(image_path):
	img = io.imread(image_path)
	h   = img.shape[0]
	w 	= img.shape[1]
	step_size=32
	for i in range(0, h, step_size):
		for j in range(0, w, step_size):
			img1 = img.copy()
			if i+step_size<h and j+step_size<w:
				rr,cc = draw.line(i,j, i,j+step_size)
				img1[rr,cc] = 1
				rr,cc = draw.line(i,j, i+step_size,j)
				img1[rr,cc] = 1
				rr,cc = draw.line(i+step_size,j, i+step_size,j+step_size)
				img1[rr,cc] = 1
				rr,cc = draw.line(i,j+step_size, i+step_size,j+step_size)
				img1[rr,cc] = 1
				ImageViewer(img1).show()

window(sys.argv[1])