from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
from ultralytics import YOLO
import logging
from starlette.concurrency import run_in_threadpool
import random

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    logging.error("Cannot open camera")
    exit()

# Initialize model
try:
    model = YOLO("weapons.pt")
except Exception as e:
    logging.error(f"Error initializing model: {str(e)}")
    model = None

# Warmup model if it's available
def warmup(model):
    if model:
        logging.info("Warming up model")
        try:   
            frame = cv2.imread("bus.jpg")
            _ = model(frame)
            logging.info("Model warmed up")
        except Exception as e:
            logging.error(f"Error during warmup: {str(e)}")
    else:
        logging.error("Model is not available. Skipping warmup.")

warmup(model)

def detect(model, frame):
    if model:
        try:
            results = model(frame)
            logging.info(results)
            draw_results(frame, results)  
            #return results
        except Exception as e:
            logging.error(f"Error during detection: {str(e)}")
            return None
    else:
        logging.error("Model is not available. Skipping detection.")
        return None

def draw_results(frame, results):
    # Check whether results are not None and non-empty
    if results and len(results) > 0:
        # Extract the Results object
        result_obj = results[0]
        
        # Log some of its attributes for debugging
        logging.info(f"Result object attributes: {dir(result_obj)}")
        
        # Check if 'boxes' attribute exists and is not None
        if hasattr(result_obj, 'boxes') and result_obj.boxes is not None:
            # Try to access the xyxy attribute
            try:
                # Accessing boxes and converting them to numpy array
                boxes = result_obj.boxes.xyxy[0].cpu().numpy()
            except AttributeError as e:
                logging.error(f"Cannot access box coordinates: {str(e)}")
                return
            
            # Log the extracted 'boxes' data
            logging.info(f"Extracted boxes: {boxes}")
            
            # If we successfully obtained boxes, draw them
            if boxes is not None:
                if len(boxes.shape) == 1:  # single box
                    xyxy, conf, cls = boxes[:4], boxes[4], boxes[5]
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)
                elif len(boxes.shape) == 2:  # multiple boxes
                    for box in boxes:
                        xyxy, conf, cls = box[:4], box[4], box[5]
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)
        else:
            logging.warning("No detection boxes found in the results.")



def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

@app.on_event("shutdown")
def shutdown_event():
    logging.info("Releasing camera")
    camera.release()

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def generate():
    while True:
        try:
            success, frame = await run_in_threadpool(camera.read)
            if not success:
                logging.error("Failed to read frame")
                break
            
            _ = await run_in_threadpool(detect, model, frame)  
            
            _, jpeg = await run_in_threadpool(cv2.imencode, '.jpg', frame)
            
            if not _:
                logging.error("Failed to encode frame")
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        except Exception as e:
            logging.error(f"An error occurred in generate(): {str(e)}")
            break

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
