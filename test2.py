import numpy as np
import cv2
import math

# input image
img = cv2.imread('image/kid.jpg')
# Convert RGB to Gray image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find maximum intensity of Grayscale
"""
smallest = np.amin(img_gray)
biggest = np.amax(img_gray)
"""
norm_imgGray = cv2.normalize(img_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

"""
print(smallest)
print(biggest)
"""
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

norm_imghsv = cv2.normalize(img_hsv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


h, s, v = cv2.split(norm_imghsv)

img_low = v - norm_imgGray
ret, thresh = cv2.threshold(img_low, 0.2, 1, cv2.THRESH_BINARY) 


"""
hsv_split = np.concatenate((h, s, v), axis=1)
"""

cv2.imshow("result", thresh)
cv2.waitKey(0)

"""
for i in range(rows):
    for j in range(cols):
        _intensity = img[i,j]
        print(_intensity)
"""
