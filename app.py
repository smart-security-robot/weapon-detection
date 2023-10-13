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

app = Flask(__name__)

# Use default camera
camera = cv2.VideoCapture(0)  

# Load YOLO model
model = YOLO("weapons.pt")

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            break
        
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
                
                # Draw box in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                print("Confidence --->", confidence)

        ret, jpeg = cv2.imencode('.jpg', frame)
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
    app.run(host='0.0.0.0', port=5000, threaded=True)
