import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\n\n\nCannot Open Webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n\n\nError Receiving Frame!")
        break

    cv2.imshow('Play - guntingbatukertas', frame)
    key = cv2.waitKey(10)
