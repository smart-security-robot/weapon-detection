from ultralytics import YOLO
import cv2
import math
from collections import defaultdict
import numpy as np
import torch

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
    
    def __init__(self, weights_path):
        self.model = None
        self.class_names = ['Grenade', 'Gun', 'Knife', 'Pistol', 'handgun', 'rifle']
        self.load_model(weights_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.warmup()
    
    def load_model(self, weights_path):
        try:
            self.model = YOLO(weights_path)
        except Exception as e:
            print(f"Error initializing model: {str(e)}")

    def warmup(self):
        if self.model:
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

    def detect_objects(self, image):
        if image is None:
            print("No image to detect objects on.")
            return None
        print("using: "+ str(self.device))
        results = self.model(image, stream=True, device=self.device)
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
                if class_name == 'Pistol':
                    color = (0, 204, 255)
                elif class_name == "Knife":
                    color = (222, 82, 175)
                elif class_name == "handgun":
                    color = (0, 149, 255)
                else:
                    color = (85, 45, 255)
                if conf > 0.5:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(image, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        return image
    
    def detect_person(self, image):
        results = self.model(image, stream=True, device=self.device, classes=0)
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
                color = (85, 45, 255)
                if conf > 0.5:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(image, (x1, y1), c2, color, -1, cv2.LINE_AA)

        return image

    
    def detect_and_track_objects(self, image):
        # Perform object detection and tracking
        results = self.model.track(image, persist=True, device=self.device)  # Note the `track` method
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
                color = self.get_color(class_name)
                if conf > 0.5:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(image, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        return image
    
    def detect_and_track_objects2(self, image):
        results = self.model.track(image, persist=True, device=self.device)

        boxes = results[0].boxes.xywh.cpu()

        #check tensor size if size (0, 4) then return image
        if boxes.size(0) == 0:
            return image
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotated_image = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track)>30:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated_image, [points], isClosed=False, color=self.get_color(self.class_names[0]), thickness=10)

            return annotated_image
        except Exception as e:
            print("No object detected")
            return image
        
    def detect_pose(self, image):
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


    # def _draw_poses(self, image, keypoints):
    #     if keypoints is None or len(keypoints) == 0:
    #         print("No keypoints to draw.")
    #         return image
        
    #     keypoints = np.array(keypoints)
    #     print(f"Keypoints Shape: {keypoints.shape}")
    #     print(f"Keypoints: {keypoints}")

    #     # Loop through all detected persons
    #     for person_keypoints in keypoints:
    #         # Draw keypoints
    #         for keypoint in person_keypoints:
    #             print(f"Processing keypoint: {keypoint}, type: {type(keypoint)}")
    #             try:
    #                 x, y = keypoint
    #                 if x > 0 and y > 0:
    #                     image = cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    #             except ValueError as e:
    #                 print(f"Failed to unpack keypoint {keypoint}: {e}. Skipping this keypoint.")
    #                 continue
            
    #         # Draw skeleton
    #         for i, (start_p, end_p) in enumerate(SKELETON):
    #             try:
    #                 x1, y1 = person_keypoints[start_p]
    #                 x2, y2 = person_keypoints[end_p]
    #             except ValueError as e:
    #                 print(f"Error unpacking keypoint: {e}.")
    #                 print(f" - Problematic keypoints: {person_keypoints[start_p]} and {person_keypoints[end_p]}.")
    #                 print(f" - Full keypoints array: {keypoints}.")
    #                 continue
                
    #             # Only draw the skeleton if both points are valid
    #             if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
    #                 image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[i], 2)
    #     return image

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