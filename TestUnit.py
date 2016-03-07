import sys
import os
import pickle
import numpy as np
from collections import defaultdict
from skimage import io
from skimage import transform
from skimage.transform import pyramid_gaussian
from test_image import Text_classification

tc = Text_classification()

# Loading the images
path = '/home/ayush/Documents/CNN/English/Fnt/Sample0'
DATA_SET_VAL = 63
# DATA_SET_VAL = 1

ans = defaultdict()
final_count = 0
for i in range(DATA_SET_VAL):
	full_path=''
	if i+1<=9:
		full_path = path + '0' + str(i+1) + '/' 
	else:
		full_path = path + str(i+1) + '/'

	count = 0
	for image_name in os.listdir(full_path):
		image_path = full_path + image_name
		tc.put_image(image_path)
		# ans[tc.predict_class()[0][0]]+=1
		if tc.predict_class()[0][0] == i:
			count+=1
	print i, count
	final_count+=count


print '-' * 20
print 'Overall Accuracy : ', (final_count*1.0)/65080 
# print tc.predict_class()
# for i in ans:
	# print i,ans[i]
# pyramids = tc.generate_image_pyramids()