import cv2
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)

#initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    logging.error("Cannot open camera")
    exit()

#initialize model
try:
    model = YOLO("weapons.pt")
except Exception as e:
    logging.error(f"Error initializing model: {str(e)}")
    model = None

#warmup model if it's available if it works return and log results
def warmup(model):
    if model:
        logging.info("Warming up model")
        try:
            
            frame = cv2.imread("bus.jpg")
            resp = model(frame)
            logging.info("Model warmed up")
            logging.info(resp)
        except Exception as e:
            logging.error(f"Error during warmup: {str(e)}")
    else:
        logging.error("Model is not available. Skipping warmup.")


warmup(model)
