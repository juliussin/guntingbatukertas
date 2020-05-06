import cv2
import numpy as np
import os
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\n\n\nCannot Open Webcam!")
    exit()

cap.set(3, 640)  # Set Webcam Width to 640
cap.set(4, 480)  # Set Webcam Height to 480
cap.set(5, 30)   # Set Webcam to 30 FPS

print('\n')
print('Welcome to Record Dataset')
print('  ~ guntingbatukertas ~  ')
print('\n')
print('  ## List of Hotkeys ##  ')
print(' - ESC: exit')
print(' - A  : capture gunting')
print(' - S  : capture batu')
print(' - D  : capture kertas')
print(' - F  : capture none (background)')

gunting_cap = os.path.join('capture', 'gunting')
batu_cap = os.path.join('capture', 'batu')
kertas_cap = os.path.join('capture', 'kertas')
none_cap = os.path.join('capture', 'none')

for directory in [gunting_cap, batu_cap, kertas_cap, none_cap]:
    if not os.path.exists(directory):
        os.makedirs(directory)

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n\n\nError Receiving Frame!")
        break
    frame = cv2.rectangle(frame,
                          (120-2, 40-2),  # Start Coordinate
                          (520+2, 440+2),  # End Coordinate
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
    frame = cv2.putText(frame,
                        'A: Gunting - S: Batu - D: Kertas - F: None',  # Text
                        (65, 470),  # Origin (bottom-left corner of the text)
                        cv2.FONT_HERSHEY_SIMPLEX,  # Font
                        0.7,  # Font Scale
                        (0, 255, 0),  # Color in BGR
                        1,  # Thickness
                        cv2.LINE_AA)  # Line Type (optional)
    crop = frame[40:440, 120:520]
    cv2.imshow('Record Dataset - guntingbatukertas', frame)
    # cv2.imshow('Crop', crop)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('a'):
        file_name = 'gunting_%s.bmp' % (str(time.time()))
        if cv2.imwrite(os.path.join(gunting_cap, file_name), crop):
            print('\nSuccessfully Save Gunting Image: ' + file_name)
        else:
            print('\nFailed to Save Gunting Image!')
    elif key == ord('s'):
        file_name = 'batu_%s.bmp' % (str(time.time()))
        if cv2.imwrite(os.path.join(batu_cap, file_name), crop):
            print('\nSuccessfully Save Batu Image: ' + file_name)
        else:
            print('\nFailed to Save Batu Image!')
    elif key == ord('d'):
        file_name = 'kertas_%s.bmp' % (str(time.time()))
        if cv2.imwrite(os.path.join(kertas_cap, file_name), crop):
            print('\nSuccessfully Save Kertas Image: ' + file_name)
        else:
            print('\nFailed to Save Kertas Image!')
    elif key == ord('f'):
        file_name = 'none_%s.bmp' % (str(time.time()))
        if cv2.imwrite(os.path.join(none_cap, file_name), crop):
            print('\nSuccessfully Save None (Background) Image: ' + file_name)
        else:
            print('\nFailed to Save None (Background) Image!')
    else:
        continue
