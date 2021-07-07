# Import the necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import the feature file for the face for Cascade classification
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the camera and define a normal sized window for output display
capturer =  cv2.VideoCapture(0)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

while True:
    # Read the frame
    ret, frame = capturer.read()

    # Convert the color space to YUV
    image_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Equalize the histogram
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

    # Convert the image back to RGB
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    # Remove the noise and convert to gray
    blur = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Threhold the image for haar cascade classification
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  

    # Detect the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For the location where the face has been determined
    for (x, y, w, h) in faces:
        # Remove the face by drawing a rectangle over the face
        frame[y:y+h, x:x+w] = [0, 0, 0]
        cv2.rectangle(image_rgb,(x, y), (x + w, y + h), (255, 0, 0), 2)
        print(x, y, x+w, y+h)

    # Show the face
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(100) & 0xFF
    if key == 27:
        print('esc')
        break
    
# Destroy the window and release the camera
cv2.waitKey(1)
capturer.release()
cv2.destroyAllWindows()
cv2.waitKey(1)