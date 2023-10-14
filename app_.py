"""
    -----------------------------------------------------------
    ECE441 - Illinois Institute of Technology
    Author: Alae Moudni
    Email: amoudni@hawk.iit.edu
    Date: October 2023
    
    Description:
    This code is a part of the ECE441 project. Unauthorized copying,
    distribution, or any form of plagiarism without explicit permission is
    strictly prohibited. All rights reserved.
    -----------------------------------------------------------
"""

##################################################

from flask import Flask, render_template, Response
import cv2
import math
from ultralytics import YOLO  # Ensure you have the ultralytics package installed
from time import sleep

app = Flask(__name__)

# Use default camera
#check if can use webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera trying second camera")
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

def load_model():
    model = YOLO("weapons.pt")
    return model

def warmup(model):
    # Warmup
    print("Warming up model")
    for i in range(10):
        print(i)
        frame = cv2.imread("bus.jpg")
        results = model(frame)
    print("Model warmed up")
    
def detect(model, frame):
    # Perform detection
    results = model(frame)
    
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = math.ceil((box.conf[0]*100))/100
            if confidence < 0.8:
                continue
            
            # # Draw box in frame
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # print("Confidence --->", confidence)
    print(results)
    return results


@app.route('/')
def index():
    # # Load model
    # model = load_model()

    # # Warmup
    # warmup(model)    


    return render_template('index.html')


def generate():
    # Load model
    model = load_model()

    # Warmup
    warmup(model)

    

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            break
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Perform detection
        results = detect(model, frame)

        if not ret:
            print("Failed to encode frame")
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    print("Starting app")

    app.run(host='0.0.0.0', port=5000, threaded=True)











# def generate():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             print("Failed to read frame")
#             break
        
#         # Perform detection
#         results = model(frame)
        
#         for r in results:
#             boxes = r.boxes
            
#             for box in boxes:
#                 # Bounding box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 confidence = math.ceil((box.conf[0]*100))/100
#                 if confidence < 0.8:
#                     continue
                
#                 # Draw box in frame
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
#                 print("Confidence --->", confidence)

#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             print("Failed to encode frame")
#             break
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')




