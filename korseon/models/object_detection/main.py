import cv2
import numpy as np
import sys, os, json, base64, time
sys.path.append(os.path.abspath("src/modules"))
from lightweight_communication_bridge import LCB

# Run object detection and return response
def onMessage(payload: dict[str, object]) -> None:
    print('received message')
    imgBytes = base64.b64decode(payload['image'])
    image = cv2.imdecode(np.frombuffer(imgBytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    print('running detection')
    result = runDetection(image)
    resized_image = cv2.resize(result[0], (640, 480))
    print('encoding result')
    _, buffer = cv2.imencode('.jpg', resized_image) # result[0]
    print('sending result')
    lcb.send({'source':'object-detection', 'code': 'result', 'image': base64.b64encode(buffer).decode('utf-8')})


# Set up LCB and register listener
lcb = LCB(onMessage, host='host.docker.internal', port='8080')
keepAlive = True

# YOLO configs
weights_path = "/yolo/yolov4.weights"
config_path = "/yolo/yolov4.cfg"
names_path = "/yolo/coco.names"

# Hyperparams
CONFIDENCE_THRESHOLD = 0.5

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Config YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def runDetection(image) -> tuple:
    height, width = image.shape[:2]

    # Preproc
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    # Generate detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                label = f"{classes[class_id]}: {confidence:.2f}"
                print(label)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return (image, detections)

def main():
    while True:
        time.sleep(1)

main()