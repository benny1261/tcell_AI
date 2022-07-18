import os
import scipy.io as sio
import cv2
import numpy as np


os.chdir('data')
raw_file = 'B1_processed.raw'
mask_file = 'B1_merged_color_mask.png'

# saving as .mat dtype ----------------------------------------------------------------------------------------------------------
# print('\ncreating matfile......')
# img = cv2.imread(mask_file, 0)
# sio.savemat('data_y.mat', {'data_y': img})

# vraw = np.fromfile(raw_file , dtype = 'float32')
# raw = vraw.reshape(img.shape[0], img.shape[1], 150) #(h, w, c)
# sio.savemat('data_x.mat', {'data_x': raw})

# loading data -------------------------------------------------------------------------------------------------------------------
x = sio.loadmat('data_x.mat')['data_x']
y = sio.loadmat('data_y.mat')['data_y']
print(x.shape)
print(y.shape)

x_ex = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
y_ex = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
print(x_ex.shape)
print(y_ex.shape)