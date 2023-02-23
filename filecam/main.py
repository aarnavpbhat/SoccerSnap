import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


# Set up the GUI
root = tk.Tk()
root.title("Gesture-based File Management System")

# Create a canvas to display the video feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()


# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Create a label to display the recognized gesture
gesture_label = tk.Label(root, text="Gesture: no_gesture", font=("Helvetica", 20))
gesture_label.pack()

# Create a label to display the file operation
file_op_label = tk.Label(root, text="File operation: ", font=("Helvetica", 20))
file_op_label.pack()

# Define a placeholder for file operation
file_operation = "No operation"

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary mask
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    contour_sizes = [(cv2.contourArea(contour), contour)
                    for contour in contours]
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Draw a bounding box around the hand
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Recognize the hand gesture based on the bounding box dimensions
    if w > 2 * h:
        gesture = 'thumbs_up'
    elif h > 2 * w:
        gesture = 'thumbs_down'
    else:
        gesture = 'no_gesture'

    # Update the GUI with the recognized gesture and file operation
    gesture_label.config(text="Gesture: " + gesture)
    file_op_label.config(text="File operation: " + file_operation)

    # Display the resulting frame on the canvas
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, image=img, anchor=tk.NW)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary mask
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    contour_sizes = [(cv2.contourArea(contour), contour)
                     for contour in contours]
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Draw a bounding box around the hand
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gesture recognition', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Start the GUI main loop
root.mainloop()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
