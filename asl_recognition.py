# Import the face_recognition
import face_recognition

# Import the necessary libraries for the code
import numpy as np
import cv2
import copy
import keras
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import cv2

# Define the simple model to import the weights trained in simple_network.py
def SimpleModel():
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(25, activation=tf.nn.softmax)
    ])
    return model

# Define the labels for Kaggle Mnist data
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Instantiate the model and load the weights
model = SimpleModel()
model.load_weights('networks/simple_nn_weights.h5')

# Load the model trained in vgg16_network.py
vgg_model = load_model('networks/vgg16_model.h5')

# Open the camera for video capture and create a window for display
capturer = cv2.VideoCapture(0)
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# THIS PART, UNTIL NEXT COMMENT, HAS BEEN DERIVED FROM 
# https://pypi.org/project/face_recognition/
# Define some random face for face recognition
user_image = face_recognition.load_image_file("face.png")
user_face_encoding = face_recognition.face_encodings(user_image)[0]
known_face_encodings = [
    user_face_encoding,
]
known_face_names = [
    "User",
]

# Initialize variables for storing the face locations
face_locations = []
face_encodings = []
face_names = []

# flag to see if the face is present or not
process_this_frame = True

# Background subtractor instantiated
backSub = cv2.createBackgroundSubtractorKNN()

# Till the camera is open
while capturer.isOpened():

    # Capture the frame
    ret, frame = capturer.read()

    # Convert the image to small resolution to avoid computation time overhead    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # FOLLOWING HAS BEEN IMPLEMENTED FROM https://pypi.org/project/face_recognition/
    # If the face present
    if process_this_frame:
        # Locate the face
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Get the face's parameteres
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            print(matches)

            if True in matches:
                index = matches.index(True)
                name = known_face_names[index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Create a copy of current frame for processing
    face_img = frame.copy()

    # Create a mask of all ones for face
    faceMask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # For the location of the face
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Bring the coordinates to original coordinate size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Pad extra pixels around the face to remove them
        faceMask[top - 50:bottom + 50, left - 50:right + 50] = 0

    # Create the face mask
    face_img = cv2.bitwise_and(face_img, face_img, mask=faceMask)
    fgMask = backSub.apply(face_img)

    # Remove the background on the face removed frame and combine the two masks
    rendered = cv2.bitwise_and(face_img, face_img, mask = fgMask)
    
    # Remove the background  from the face mask & apply it on the frame
    bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
    fgmask = bgModel.apply(face_img)

    # Erode the image mask to remove clutter
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    img = cv2.bitwise_and(face_img, face_img, mask=fgmask)

    # Convert the image to HSV for color range based hand isolation
    img = face_img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # Get the image that is within the color range
    skinMask = cv2.inRange(hsv, lower, upper)

    # Use the mask on the frame
    img = cv2.bitwise_and(img, img, mask=skinMask)
    skinMask1 = copy.deepcopy(skinMask)

    # Find contour on the color range detected objects
    _,contours, hierarchy = cv2.findContours(skinMask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1

    # Find the region with maximum area and classify it as the hand
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = img
        handMask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(handMask, [res], -1, (255, 255, 255), -1)
        handMask = cv2.cvtColor(handMask, cv2.COLOR_BGR2GRAY)
        finalImage = cv2.bitwise_and(frame, frame, mask=handMask)

        # Reshape the final hand image to input size of the network
        reshaped_image = cv2.resize(finalImage,(200, 200))
        reshaped_image = reshaped_image.reshape((1, 200, 200, 3))

        # Apply the image to the network and display the prediction on the output
        pred = vgg_model.predict(reshaped_image)
        cv2.putText(img = finalImage, text = str(labels[pred.argmax() - 1]), org = (120, 120), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 5, color = (0, 255, 0))
        cv2.imshow('Video', finalImage)

    key = cv2.waitKey(100) & 0xFF
    if key == 27:
        print('esc')
        break

# destroy all windows and release the camera
capturer.release()
cv2.destroyAllWindows()