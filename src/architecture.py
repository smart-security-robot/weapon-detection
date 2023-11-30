from ultralytics import YOLO
import cv2
import math
from collections import defaultdict
import numpy as np
import torch
from core.settings import settings
from starlette.concurrency import run_in_threadpool
import time
from src.incident import send_incident, get_all_users
import base64
import io
from PIL import Image
import face_recognition
import datetime

track_history = defaultdict(lambda: [])

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # Head
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), # Arms
    (5, 11), (6, 12), (11, 12),             # Shoulders and Hip
    (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]


COLORS = [
    (0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0),
    (255, 127, 0), (255, 0, 0), (255, 0, 127), (255, 0, 255),
    (127, 0, 255), (0, 0, 255), (127, 255, 0), (255, 255, 0),
    (0, 127, 255), (0, 255, 127), (127, 0, 255), (255, 127, 0)
]



class YOLOModel:
    
    def __init__(self):
        self.model = None
        self.person_model = YOLO(settings.person_detection_weight)
        self.weapon_model = YOLO(settings.weapon_detection_weight)
        self.pose_estimation = YOLO(settings.pose_estimation_weight)
        self.class_names = ['Grenade', 'Gun', 'Knife', 'Pistol', 'handgun', 'rifle']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.warmup_person()
        #self.load_model(settings.weapon_detection_weight)
        self.warmup_gen(self.model)
        self.warmup_weapon()
        self.warmup_person()
        self.warmup_pose()
        self.person_bbox_margin = settings.person_bbox_margin
        self.person_results = False
        self.weapon_results = False
        self.ALERT_THROTTLE_TIME = 3
        self.alert_timestamps = defaultdict(lambda: 0)

    
    def load_model(self, weights_path):
        try:
            self.model = YOLO(weights_path)
        except Exception as e:
            print(f"Error initializing model: {str(e)}")

    def warmup_person(self):
        if self.person_model:
            print("Warming up person detection model")
            try:   
                frame = cv2.imread("test.jpg")
                _ = self.person_model(frame)
                print("Person Detection Model warmed up")
            except Exception as e:
                print(f"Error during warmup: {str(e)}")
                exit()
        else:
            print("Model is not available. Skipping warmup.")

    def warmup_weapon(self):
        if self.weapon_model:
            print("Warming up weapon detection model")
            try:   
                frame = cv2.imread("test.jpg")
                _ = self.weapon_model(frame)
                print("Weapon Detection Model warmed up")
            except Exception as e:
                print(f"Error during warmup: {str(e)}")
                exit()
        else:
            print("Model is not available. Skipping warmup.")

    def warmup_pose(self):
        if self.pose_estimation:
            print("Warming up pose estimation model")
            try:   
                frame = cv2.imread("test.jpg")
                _ = self.pose_estimation(frame)
                print("Pose Estimation Model warmed up")
            except Exception as e:
                print(f"Error during warmup: {str(e)}")
                exit()
        else:
            print("Model is not available. Skipping warmup.")

    def warmup_gen(self, model):
        if model:
            print("Warming up model")
            try:   
                frame = cv2.imread("test.jpg")
                _ = self.model(frame)
                print("Model warmed up")
            except Exception as e:
                print(f"Error during warmup: {str(e)}")
                exit()
        else:
            print("Model is not available. Skipping warmup.")



    def get_color(self, class_name):
        # Define color code based on class_name
        if class_name == 'Pistol':
            return (0, 204, 255)
        elif class_name == "Knife":
            return (222, 82, 175)
        elif class_name == "handgun":
            return (0, 149, 255)
        else:
            return (85, 45, 255)

    @staticmethod    
    def adjust_bounding_box(x1, y1, x2, y2, width, height, margin):
        x_margin = int((x2 - x1) * margin)
        y_margin = int((y2 - y1) * margin)

        x1 = max(0, x1 - x_margin)
        y1 = max(0, y1 - y_margin)
        x2 = min(width, x2 + x_margin)
        y2 = min(height, y2 + y_margin)

        x1 = min(x1, x2-1)
        y1 = min(y1, y2-1)

        return x1, y1, x2, y2

##############################################################################################################
#Pose
##############################################################################################################
    def detect_pose_and_draw(self, image):
        #self.model.eval()  # Ensure the model is in evaluation mode

        # Convert image to suitable format
        input_tensor = self._prepare_image(image)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(input_tensor, device=self.device)

        # Process outputs 
        keypoints = self._process_outputs(outputs, image.shape)

        # Draw keypoints and skeletons on the image
        annotated_image = self._draw_poses(image, keypoints)

        return annotated_image

    def _prepare_image(self, image):
        # Resize the image
        image = cv2.resize(image, (640, 640))

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert the image to PyTorch tensor
        input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()

        # Add a batch dimension and normalize
        input_tensor = input_tensor.unsqueeze(0) / 255.0  

        # Move tensor to the appropriate device
        input_tensor = input_tensor.to(self.device)  

        return input_tensor

    def _process_outputs(self, outputs, original_shape):
        # Ensure outputs is a numpy array, if it's a list perhaps take an element.
        if isinstance(outputs, list):
            outputs = outputs[0].keypoints.xy
        keypoints = outputs.cpu().numpy().squeeze()
        
        # Handling all zero keypoints
        if np.all(keypoints == 0):
            print("No keypoints detected.")
            return None
        return keypoints

    def _draw_poses(self, image, keypoints):
        if keypoints is None or len(keypoints) == 0:
            print("No keypoints to draw.")
            return image
        
        keypoints = np.array(keypoints)
        print(f"Keypoints Shape: {keypoints.shape}")
        print(f"Keypoints: {keypoints}")

        # If keypoints array is 3-dimensional, treat it as [person][keypoint][coordinates]
        if len(keypoints.shape) == 3:
            for person_keypoints in keypoints:
                image = self._draw_single_person(image, person_keypoints)
        # If keypoints array is 2-dimensional, treat it as [keypoint][coordinates]
        elif len(keypoints.shape) == 2:
            image = self._draw_single_person(image, keypoints)
        else:
            print("Unexpected keypoints shape, skipping drawing.")
        
        return image
    
    def _draw_single_person(self, image, keypoints):
        original_height, original_width = image.shape[:2]
        resized_height, resized_width = (640, 640)

        width_scale = original_width / resized_width
        height_scale = original_height / resized_height

        # Draw keypoints
        for keypoint in keypoints:
            print(f"Processing keypoint: {keypoint}, type: {type(keypoint)}")
            try:
                x, y = keypoint
                x = int(x * width_scale)
                y = int(y * height_scale)

                if x > 0 and y > 0:
                    image = cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
            except ValueError as e:
                print(f"Failed to unpack keypoint {keypoint}: {e}. Skipping this keypoint.")
                continue
        
        # Draw skeleton
        for i, (start_p, end_p) in enumerate(SKELETON):
            try:
                x1, y1 = keypoints[start_p]
                x2, y2 = keypoints[end_p]

                x1 = int(x1 * width_scale)
                y1 = int(y1 * height_scale)
                x2 = int(x2 * width_scale)
                y2 = int(y2 * height_scale)

            except ValueError as e:
                print(f"Error unpacking keypoint: {e}.")
                print(f" - Problematic keypoints: {keypoints[start_p]} and {keypoints[end_p]}.")
                print(f" - Full keypoints array: {keypoints}.")
                continue
            
            # Only draw the skeleton if both points are valid
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[i], 2)

        return image


##############################################################################################################
#Detect and draw
##############################################################################################################
    async def detect_weapon_and_draw(self, image):
        if image.size == 0:
            print("No image to detect objects on.")
            return False
        
        results = self.weapon_model(image, stream=True, device=self.device)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                result = (class_name, conf)

                if class_name == 'Pistol':
                    color = (0, 204, 255)
                elif class_name == "Knife":
                    color = (222, 82, 175)
                elif class_name == "handgun":
                    color = (0, 149, 255)
                else:
                    color = (85, 45, 255)

                if conf > settings.weapons_detection_thresh:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(image, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                    return result, image

        return False, image    
    
    async def detect_and_track_person_and_draw(self, image):
        results = self.person_model.track(image, persist=True, device=self.device)

        # Check if results are valid
        if not results or len(results) == 0 or results[0] is None or results[0].boxes is None:
            print(f"Results: {results}")
            return False

        boxes = results[0].boxes.xywh.cpu()

        #check tensor size if size (0, 4) then return image
        if boxes.size(0) == 0:
            return image
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidence = results[0].boxes.conf.cpu().tolist()
            detections = []
            annotated_image = results[0].plot()

            for box, track_id, conf in zip(boxes, track_ids, confidence):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track)>30:
                    track.pop(0)


                bounding_box = box.int().cpu().tolist()
                detections.append((track_id, bounding_box, conf))
                points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated_image, [points], isClosed=False, color=self.get_color(self.class_names[0]), thickness=10)

            return detections, annotated_image
        except Exception as e:
            print("No object detected")
            return image

    async def face_recognition_and_draw_users(image, person_bbx):
        # Convert to RGB since face_recognition uses RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face encodings
        face_encodings = face_recognition.face_encodings(rgb_image)
        if len(face_encodings) == 0:
            print("No face found")
            return False, image

        # Retrieve users
        users = get_all_users()
        if users is None or not isinstance(users, list):
            print("No users found")
            return False, image

        # Prepare users' face data
        valid_users = [user for user in users if user['face_vector'] is not None]
        face_encodings_users = [np.array(user['face_vector']) for user in valid_users]
        first_names = [user['first_name'] for user in users if user['face_vector'] is not None]
        last_names = [user['last_name'] for user in users if user['face_vector'] is not None]

        # Find the best match for the detected face
        distances = face_recognition.face_distance(face_encodings_users, face_encodings[0])
        best_match_index = np.argmin(distances)
        best_match_distance = distances[best_match_index]
        if best_match_distance > 0.4:
            return False, image

        # Get the name of the best match
        name = first_names[best_match_index] + " " + last_names[best_match_index]

        # Draw facial landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_image)
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                points = face_landmarks[facial_feature]
                for point in points:
                    cv2.circle(image, point, 2, (0, 255, 0), -1)

        return name, image


    async def pipeline_and_draw(self, image):
        # Step 1: Detect and track persons
        person_results, annotated_image = await self.detect_and_track_person_and_draw(image)

        if not person_results:
            return False, image
        
        detected_weapons = []
        faces_recognized = []
        person_processed = []

        for track_id, person_bbx, conf in person_results:

            if conf < settings.person_detection_thresh:
                print("Person confidence is too low")
                break

            # if track_id not in faces_recognized:
            #     #only process each person once
            #     faces_recognized.append(track_id)
            #     #recognize faces
            #     name = self.face_recognition_users(image, person_bbx)
            #     if name:
            #         person_processed.append((track_id, person_bbx, name))
            # else:
            #     continue

            # Step 3: Detect weapons in the person ROI
            weapon_results, annotated_image = await self.detect_weapon_and_draw(annotated_image)

            if weapon_results:
                # Step 4: Check for new detection and send alert if necessary
                #check if track_id is in alert_timestamps
                if track_id in self.alert_timestamps:
                    last_alert_time = self.alert_timestamps.get(track_id, 0)
                    current_time = time.time()

                else:
                    last_alert_time = 0
                    current_time = time.time()

                if current_time - last_alert_time > self.ALERT_THROTTLE_TIME:
                    self.alert_timestamps[track_id] = current_time
                    detected_weapons.append((track_id, weapon_results))

                    name, annotated_image = await self.face_recognition_and_draw_users(annotated_image, person_bbx)
                    if name:
                        person_name = name
                    else:
                        person_name = "Unknown"

                    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(image_rgb)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    #get label of detected weapon
                    label = weapon_results[0]
                    current_time = datetime.datetime.now().isoformat()
                    send_incident("Detected: "+label, current_time, img_str, "location", person_name)

        return detected_weapons if detected_weapons else False, annotated_image

##############################################################################################################
#Detect pipeline
##############################################################################################################


    async def detect_objects(self, image, person_bbx):
        if image.size == 0:
            print("No image to detect objects on.")
            return False
        # x1, y1, x2, y2 = [int(x) for x in person_bbx]

        # height, width, _ = image.shape

        # x1, y1, x2, y2 = self.adjust_bounding_box(x1, y1, x2, y2, width, height, self.person_bbox_margin)

        # if x2 - x1 <= 0 or y2 - y1 <= 0:
        #     print(f"Invalid person bounding box in detect objects: {x1, y1, x2, y2}")
        #     return False

        # image = image[y1:y2, x1:x2]

        results = self.weapon_model(image, stream=True, device=self.device)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                result = (class_name, conf)
                if conf >= settings.weapons_detection_thresh:
                    return result
        print(f"results: {results}")
        return False            

    async def detect_and_track_person(self, image):
        results = self.person_model.track(image, persist=True, device=self.device, classes=0)

        # Check if results are valid
        if not results or len(results) == 0 or results[0] is None or results[0].boxes is None:
            print(f"Results: {results}")
            return False
        
        #print(f"Results: {results}")

        boxes = results[0].boxes.xywh.cpu()

        # Check tensor size; if size (0, 4), then return False
        if boxes.size(0) == 0:
            return False

        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()  # Assuming there is a 'conf' attribute
            detections = []

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                bounding_box = box.int().cpu().tolist()
                detections.append((track_id, bounding_box, conf))
            print(f"Detections: {detections}")
            return detections
        
        except Exception as e:
            print(f"Exception in detect_and_track_person: {str(e)}")
            return False
    
    
    def face_recognition_users(self, image, person_bbx):
        # x1, y1, x2, y2 = [int(x) for x in person_bbx]

        # height, width, _ = image.shape

        # x1, y1, x2, y2 = self.adjust_bounding_box(x1, y1, x2, y2, width, height, self.person_bbox_margin)

        # if x2 - x1 <= 0 or y2 - y1 <= 0:
        #     print(f"Invalid person bounding box in face recognition: {x1, y1, x2, y2}")
        #     return False
        
        # image = image[y1:y2, x1:x2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) == 0:
            print("No face found")
            return False
        else:
            #send request to get all users
            users = get_all_users()

            if users is None or not isinstance(users, list):
                print("No users found")
                return False
            
            valid_users = [user for user in users if user['face_vector'] is not None]
            #get all face_encodings from users also append first name and last name
            face_encodings_users = [np.array(user['face_vector']) for user in valid_users]
            #get all first names and last names
            first_names = [user['first_name'] for user in users if user['face_vector'] is not None]
            last_names = [user['last_name'] for user in users if user['face_vector'] is not None]
            distances = face_recognition.face_distance(face_encodings_users, face_encodings[0])
            best_match_index = np.argmin(distances)
            best_match_distance = distances[best_match_index]
            if best_match_distance > 0.4:            
                return False
            else:
                #concatenate first name and last name
                name = first_names[best_match_index] + " " + last_names[best_match_index]
                return name    
    
    
    async def pipeline(self, image):
        # Step 1: Detect and track persons
        person_results = await self.detect_and_track_person(image)

        if not person_results:
            return False
        
        detected_weapons = []
        faces_recognized = []
        person_processed = []

        for track_id, person_bbx, conf in person_results:

            if conf < settings.person_detection_thresh:
                print("Person confidence is too low")
                break

            # if track_id not in faces_recognized:
            #     #only process each person once
            #     faces_recognized.append(track_id)
            #     #recognize faces
            #     name = self.face_recognition_users(image, person_bbx)
            #     if name:
            #         person_processed.append((track_id, person_bbx, name))
            # else:
            #     continue

            # Step 3: Detect weapons in the person ROI
            weapon_results = await self.detect_objects(image, person_bbx)

            if weapon_results:
                # Step 4: Check for new detection and send alert if necessary
                #check if track_id is in alert_timestamps
                if track_id in self.alert_timestamps:
                    last_alert_time = self.alert_timestamps.get(track_id, 0)
                    current_time = time.time()

                else:
                    last_alert_time = 0
                    current_time = time.time()

                if current_time - last_alert_time > self.ALERT_THROTTLE_TIME:
                    self.alert_timestamps[track_id] = current_time  # Update the last alert time
                    detected_weapons.append((track_id, weapon_results))
                    #send incident report
                    #convert image to base64 then send
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(image_rgb)  # Convert to PIL Image object
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    name = self.face_recognition_users(image, person_bbx)

                    if name:
                        person_name = name
                    else:
                        person_name = "Unknown"
                    #get person name
                    #person_name = [person[2] for person in person_processed if person[0] == track_id][0]
                    #matching_person = next((person for person in person_processed if person[0] == track_id), None)

                    #if person name is not in the list then send unknown
                    # if matching_person:
                    #     person_name = matching_person[2]
                    # else:
                    #     person_name = "Unknown"
                    #send incident report
                    
                    #get label of detected weapon
                    label = weapon_results[0]
                    current_time = datetime.datetime.now().isoformat()
                    send_incident("Detected: "+label, current_time, img_str, "location", person_name)

            


        return detected_weapons if detected_weapons else False
    
        

