from ultralytics import YOLO
#load model of yolo that is trained on COCO dataset 
model=YOLO("yolov8n.pt")

def detect_person(frame):
    "returns True if person present, False if none"
    #yolo process image and outputs bounding boxes, class labels, confidence scores
    results=model(frame,verbose=False)[0]
    for c in results.boxes.cls:
        if int(c)==0:
            return True
    return False
