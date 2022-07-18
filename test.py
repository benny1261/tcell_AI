import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf

# print(tf.test.is_built_with_cuda())
# print(tf.test.is_built_with_gpu_support())

if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")