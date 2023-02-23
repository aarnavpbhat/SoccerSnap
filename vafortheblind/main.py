import cv2
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the object detection model
model = cv2.dnn.readNetFromTensorflow(
    "frozen_inference_graph.pb", "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.pbtxt")

# Load the class labels
with open("coco_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to improve performance
    frame = cv2.resize(frame, (300, 300))

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(
        frame, size=(300, 300), swapRB=True, crop=False)

    # Pass the blob through the model
    model.setInput(blob)
    output = model.forward()

    # Loop over the detections
    for i in range(output.shape[2]):
        # Get the confidence score
        confidence = output[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the class label
            class_id = int(output[0, 0, i, 1])

            # Get the bounding box coordinates
            x1 = int(output[0, 0, i, 3] * frame.shape[1])
            y1 = int(output[0, 0, i, 4] * frame.shape[0])
            x2 = int(output[0, 0, i, 5] * frame.shape[1])
            y2 = int(output[0, 0, i, 6] * frame.shape[0])

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get the class label
            label = labels[class_id]

            # Speak the class label
            engine.say(label)
            engine.runAndWait()

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
