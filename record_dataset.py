import cv2
import numpy as np
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\n\n\nCannot Open Webcam!")
    exit()

print('Welcome to Record Dataset - guntingbatukertas')
print('-- List of Hotkey --')
print('ESC: exit')
print('A: capture gunting')
print('S: capture batu')
print('D: capture kertas')

gunting_cap = 'capture/gunting/'
batu_cap = 'capture/batu/'
kertas_cap = 'capture/kertas/'

for directory in [gunting_cap, batu_cap, kertas_cap]:
    if not os.path.exists(directory):
        os.makedirs(directory)

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n\n\nError Receiving Frame!")
        break

    cv2.imshow('Record Dataset - guntingbatukertas', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord('a'):
        file_name = 'gunting_%s.bmp' % (str(datetime.now()))
        cv2.imwrite(gunting_cap + file_name, frame)
        print('Successfully Save Gunting Image: ' + file_name)
    elif key == ord('s'):
        file_name = 'batu_%s.bmp' % (str(datetime.now()))
        cv2.imwrite(batu_cap + file_name, frame)
        print('Successfully Save Batu Image: ' + file_name)
    elif key == ord('d'):
        file_name = 'kertas_%s.bmp' % (str(datetime.now()))
        cv2.imwrite(kertas_cap + file_name, frame)
        print('Successfully Save Kertas Image: ' + file_name)
    else:
        continue
