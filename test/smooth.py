from cv2 import cv2 
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


img = cv2.imread('image/kid.png')
plt.imshow(img, interpolation='nearest')
plt.show()
# Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
img = ndimage.gaussian_filter(img, sigma=(10, 10, 0), order=0)
plt.imshow(img, interpolation='nearest')
plt.show()