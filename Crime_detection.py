import cv2
import numpy as np

# Load Yolo
weightsPath = "C:\\Users\\Varalakshmi\\Downloads\\yolo-object-detection\\yolo-coco\\yolov3.weights"
configPath = "C:\\Users\\Varalakshmi\\Downloads\\yolo-object-detection\\yolo-coco\\yolov3.cfg"
labelsPath = "C:\\Users\\Varalakshmi\\Downloads\\yolo-object-detection\\yolo-coco\\coco.names"

# Load the COCO class labels from your "coco.names" file
LABELS = open(labelsPath).read().strip().split("\n")

# Replace "Knife" with "knife" to match your label in the "coco.names" file
knife_label = "knife"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get the names of the output layers from YOLO
layer_names = net.getUnconnectedOutLayersNames()

# Initialize the video stream
vs = cv2.VideoCapture("C:\\Users\\Varalakshmi\\Downloads\\yolo-object-detection\\v6.mp4")  # Use 0 for webcam or specify your video file

crime_detected = False  # Flag to track if a crime is detected

while True:
    # Read the next frame from the video stream
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, we've reached the end of the stream
    if not grabbed:
        break

    # Perform object detection with YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    # Initialize lists for detected objects and their confidences
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5 and LABELS[classID] == knife_label:
                # Scale the bounding box coordinates back to the size of the image
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype("int")

                # Append the detected object information to the lists
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Check if a knife is detected
    if len(idxs) > 0:
        crime_detected = True

    # Display "Crime detected: A person with knife" if a crime is detected
    if crime_detected:
        for i in range(len(boxes)):
            if i in idxs:
                (x, y, w, h) = boxes[i]
                color = (0, 0, 255)  # Red color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        alert_message = "Crime detected: A person with knife"
        cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Frame", frame)

    # Check for the 'q' key to quit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources and close windows
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()




