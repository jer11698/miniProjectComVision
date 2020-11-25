import imutils
import numpy as np
import argparse
import cv2

img = cv2.imread('image/acneFace.jpg')

lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(skinMask, kernel, iterations=2)
skinMask = cv2.dilate(skinMask, kernel, iterations=2)
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(img, img, mask=skinMask)

cv2.imshow("images", np.hstack([img, skin]))

# if the 'q' key is pressed, stop the loop
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()

