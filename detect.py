from ultralytics import YOLO
import cv2
import math 

# Start webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO("weapons.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
            confidence = math.ceil((box.conf[0]*100))/100
            if confidence < 0.8:
                continue

            # Put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
