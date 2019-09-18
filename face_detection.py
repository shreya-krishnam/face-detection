# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:35:37 2019

@author: Shreya
"""

import cv2
import sys


detect_img_path = sys.argv[1]
cascade_path = sys.argv[2]

#make the cascade classifier

cascade_classifier = cv2.CascadeClassifier(cascade_path)
read_image = cv2.imread(detect_img_path)
#convert the image into grayscale

gray_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)

faces = cascade_classifier.detectMultiScale(
    gray_image,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(30, 30)
)
if len(faces!=0):
    print("Found {0} faces!".format(len(faces)))
else:
    print("no faces found")

for (x, y, w, h) in faces:
    cv2.rectangle(gray_image, (x, y), (x+w, y+h), (255,255,255), 2)
    
cv2.imshow("Faces found", gray_image)
cv2.waitKey(0)
    

