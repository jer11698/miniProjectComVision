from cv2 import cv2 
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage

img_rgb = cv2.imread('image/kid.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('image/kid_detect.jpg', 0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where( res >= threshold)

print(loc[::-1])

for pt in zip(*loc[::-1]):
    #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
    crop_img = img_rgb[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
    crop_img = ndimage.gaussian_filter(crop_img, sigma=(10, 10, 0), order=0)
    img_rgb[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = crop_img


cv2.imshow('res', img_rgb)
#cv2.imwrite('impostermatch.jpeg', img_rgb)
cv2.waitKey(0)