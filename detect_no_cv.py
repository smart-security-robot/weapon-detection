from ultralytics import YOLO
import math 
import imageio

# Model
model = YOLO("weapons.pt")

# Start webcam
reader = imageio.get_reader('<video0>')  # '<video0>' refers to the default camera. Adjust as needed.

for img in reader:
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

            # Print results
            print(f"Bounding Box: [{x1}, {y1}, {x2}, {y2}]")
            print("Confidence --->", confidence)

    # Exit loop if 'q' is pressed but keep looping otherwise
    if KeyboardInterrupt:
        break


reader.close()
