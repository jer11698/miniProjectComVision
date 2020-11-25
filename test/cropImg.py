import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('image/w644.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces)

# Drawing rectangle
for (x, y, w, h) in faces:
    crop_img = img[y:y+h, x:x+w]

cv2.imshow('img', crop_img)
k = cv2.waitKey(0) & 0xff
if k == 27:
    cv2.destroyAllWindows()
