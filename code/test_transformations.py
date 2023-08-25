import cv2
import numpy as np
import h5py
f = h5py.File('../data/DRIVE/converted_data/21_training.h5', 'r')
dset = f['label']
data = np.array(dset[:,:])
file = 'test.jpg'
cv2.imwrite(file, data)