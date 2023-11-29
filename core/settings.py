
class Settings():
    weapons_detection_thresh = 0.8
    person_detection_thresh = 0.6
    threat_detection_thresh = 0.9

    # Path to the YOLO weights and model configuration
    threat_detection_weight = "src/weights/threat_detection.h5"
    person_detection_weight = "src/weights/yolov8s.pt"
    weapon_detection_weight = "src/weights/weapons_100_s.pt"
    pose_estimation_weight = "src/weights/yolov8s-pose.pt"
    person_bbox_margin = 0.2
    server_api_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGFlQGFsYWUuY29tIiwiZXhwIjoxNzM3MjEyNjMyfQ.jKHS_31dQ_fQAq_GazoDWV6eAJWpDURkFGJ2nmrMVUA"
    server_url_incidents = "http://localhost:8000/incidents/incidents/"
    server_url_users = "http://localhost:8000/users/users/"


settings = Settings()