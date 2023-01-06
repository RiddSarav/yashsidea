import cv2
import numpy as np
import pyautogui

# Load the Haar cascade for detecting hands
hand_cascade = cv2.CascadeClassifier('hand.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the scaling factor for the images
scale_factor = 0.3

# Set the window size for the mouse movement
window_size = 50

while True:
    # Read the frame
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray)

    # Iterate over the detected hands
    for (x,y,w,h) in hands:
        # Draw a rectangle around the hand
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Get the center of the hand
        cx = x + w//2
        cy = y + h//2

        # Move the mouse to the center of the hand
        pyautogui.moveTo(cx*(1/scale_factor), cy*(1/scale_factor))

    # Display the frame
    cv2.imshow('Hand Detector', frame)

    # Check for user input
    c = cv2.waitKey(1)
    if c == 27: # Esc key
        break

# Release the webcam
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()
