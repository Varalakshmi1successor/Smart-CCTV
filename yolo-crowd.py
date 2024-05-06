# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Define the class IDs for persons and trains
person_class_id = 0  # You may need to verify the correct class ID for "person" in your COCO dataset
train_class_id = 123 # Find the correct class ID for "train" in your COCO dataset

# load the COCO class labels our YOLO model was trained on
labelsPath = "C:\\Users\\Varalakshmi\\Downloads\\CROWD DETECTION\\yolo-coco\\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "C:\\Users\\Varalakshmi\\Downloads\\CROWD DETECTION\\yolo-coco\\yolov3.weights"
configPath = "C:\\Users\\Varalakshmi\\Downloads\\CROWD DETECTION\\yolo-coco\\yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading CSR-NET from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# Get the names of the output layers from YOLO
layer_names = net.getUnconnectedOutLayersNames()

# initialize the video stream, pointer to output video file, and
# frame dimensions
if args["input"] is None:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture(args["input"])
(W, H) = (None, None)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layer_names)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Check if the detected class is "person" or "train"
            if classID == person_class_id or classID == train_class_id:
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    # Initialize counts for persons and trains
    person_count = 0

    # Initialize an alert message
    alert_message = ""

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # Check if the detected object is a person and increment the count
        if classIDs[i] == person_class_id:
            person_count += 1
            # Get the center coordinates of the detected person
            centerX, centerY = boxes[i][0] + boxes[i][2] // 2, boxes[i][1] + boxes[i][3] // 2
            # Draw a small green dot at the center
            cv2.circle(frame, (centerX, centerY), 2, (0, 255, 0), -1)

    # Check if 20 persons or more are detected
    if person_count >= 20:
        alert_message = "Overcrowding Detected!"

    # Display the count of detected persons in blue if count exceeds 20
    if person_count >= 20:
        cv2.putText(frame, "Persons: {}".format(person_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Persons: {}".format(person_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the alert message directly on the frame
    if alert_message:
        cv2.putText(frame, alert_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)

    # Check if the 'q' key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# release the file pointers
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
