import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# Load the model
model = load_model('violence_detection.h5')

# Define the video path; you can use a file or a camera index
video_path = 'test-v.mp4'

# Define a function to preprocess the video frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = preprocess_input(frame)
    return frame

# Open the video file or capture device
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if the video ends or there's an issue
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Expand dimensions to match the model's expected input
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    # Perform inference
    predictions = model.predict(preprocessed_frame)
    
    # Display the predictions (you might want to further process the output)
    print(f"Predictions: {predictions}")
    
    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
