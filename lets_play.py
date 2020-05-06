import cv2
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

h5name = input('\n\n\nInput H5 File Name: ')
if h5name[-3:] != '.h5':
    h5name = h5name + '.h5'
try:
    model = tf.keras.models.load_model(h5name)
except Exception as e:
    print('\n\n\nError!')
    print(e)
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\n\n\nCannot Open Webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n\n\nError Receiving Frame!")
        break
    frame = cv2.rectangle(frame,
                          (120 - 2, 40 - 2),  # Start Coordinate
                          (520 + 2, 440 + 2),  # End Coordinate
                          (0, 255, 0),  # Color in BGR
                          1)  # Thickness
    frame = cv2.putText(frame,
                        '~ guntingbatukertas ~',  # Text
                        (140, 27),  # Origin (bottom-left corner of the text)
                        cv2.FONT_HERSHEY_SIMPLEX,  # Font
                        1,  # Font Scale
                        (0, 255, 0),  # Color in BGR
                        1,  # Thickness
                        cv2.LINE_AA)  # Line Type (optional)
    crop = cv2.cvtColor(frame[40:440, 120:520], cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (150, 150))
    print(crop.shape)
    x = np.expand_dims(crop, axis=0)
    images = np.vstack([x])
    prediction = model.predict(images.astype(np.float32), batch_size=10)
    print(prediction)
    cv2.imshow('Play - guntingbatukertas', frame)
    key = cv2.waitKey(1)
