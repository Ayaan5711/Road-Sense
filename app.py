import webbrowser
import uvicorn
import threading
import cv2
import time
from collections import defaultdict, deque
from src.video_utils import video_manifest_extractor
from src.zone_utils import create_zones
from src.tracking_utils import initialize_tracking
from src.processing import process_frame
from src.transform_utils import ViewTransformer
from ultralytics import YOLO
import supervision as sv

import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import random
from threading import Lock
import logging

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import os


# === Logging Configuration ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("roadsense.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Shared live data structure to be updated in real-time
live_data = {
    "traffic_stats": [],
    "alerts": {
        "overspeeding": [],
        "stopped_vehicles": [],
        "proximity_alerts": [],
        "accidents": []
    },
    "accident_detection": [],
    "analytics": {
        "vehicle_count_over_time": {"labels": [], "data": []},
        "speed_distribution": {"ranges": [], "counts": []},
        "vehicle_type_distribution": {"labels": [], "data": []},
        "zone_congestion": {"zones": [], "congestion_levels": []},
        "average_delay": {"labels": [], "data": []}
    }
}

live_data_lock = Lock()
camera_mode = {"mode": "live"}  # Default mode
camera_mode_lock = Lock()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

VEHICLE_MODEL_PATH = "models/yolov8s.pt"
VEHICLE_MODEL = YOLO(VEHICLE_MODEL_PATH)

colors = sv.ColorPalette.LEGACY
video_info = sv.VideoInfo.from_video_path("videos/recorded_stream_large.mp4")  # Dummy path for init
coef = video_info.width / 1280

x1 = [-160, -25, 971]; y1 = [405, 710, 671]
x2 = [112, 568, 1480]; y2 = [503, 710, 671]
x3 = [557, 706, 874];  y3 = [195, 212, 212]
x4 = [411, 569, 749];  y4 = [195, 212, 212]

SOURCES, TARGETS, zone_annotators, box_annotators, trace_annotators, line_zones, \
    line_zone_annotators, label_annotators, lines_start, lines_end, zones, \
    x1, y1, x2, y2, x3, y3, x4, y4 = create_zones(x1, y1, x2, y2, x3, y3, x4, y4, coef, video_info, colors)

byte_tracker, fps_monitor = initialize_tracking(video_info)
view_transformers = [ViewTransformer(s, t) for s, t in zip(SOURCES, TARGETS)]
selected_classes = [2, 3, 5, 7]
coordinates = [defaultdict(lambda: deque(maxlen=30)) for _ in range(3)]
fps = video_info.fps


ANN_MODEL_PATH = "models/model_epoch_50.pth"
SAVE_SNAPSHOTS = True
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load DenseNet-201 (feature extractor)
densenet = models.densenet201(pretrained=True)
feature_extractor = nn.Sequential(
    densenet.features,
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
).to(DEVICE)
feature_extractor.eval()

# ANN Classifier
class ANNClassifier(nn.Module):
    def __init__(self, input_dim=1920):
        super(ANNClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

classifier = ANNClassifier().to(DEVICE)
classifier.load_state_dict(torch.load(ANN_MODEL_PATH, map_location=DEVICE))
classifier.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# IoU-based Collision Detection
def aabb_collision(box1, box2, iou_threshold=IOU_THRESHOLD):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    if inter_area == 0:
        return False

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area

    return 0.8 > iou > iou_threshold

# Save cropped accident snapshot
def save_snapshot(crop, frame_id):
    if SAVE_SNAPSHOTS:
        os.makedirs("snapshots", exist_ok=True)
        filename = f"snapshots/accident_crop_{frame_id}.jpg"
        cv2.imwrite(filename, crop)
        print(f"[INFO] Accident snapshot saved: {filename}")






def get_video_source():
    with camera_mode_lock:
        mode = camera_mode["mode"]
    if mode == "live":
        source_url = "https://youtu.be/z545k7Tcb5o"
        return video_manifest_extractor(source_url) ,"live"
    elif mode == "static":
        return "videos/recorded_stream_large.mp4","static"
    elif mode == "accident":
        return "videos/accident_t2.mp4","accident"
    else:
        raise ValueError("Invalid camera mode selected.")

def generate_frames():
    video_path,mode = get_video_source()
    if mode == "accident":
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                results = VEHICLE_MODEL(frame)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()

                vehicles = []
                for box, cls_id, conf in zip(boxes, classes, confs):
                    if conf > CONFIDENCE_THRESHOLD:
                        vehicles.append(box.astype(int))

                # Collision detection
                for i in range(len(vehicles)):
                    for j in range(i + 1, len(vehicles)):
                        box1, box2 = vehicles[i], vehicles[j]
                        if aabb_collision(box1, box2):
                            x_min = max(0, min(box1[0], box2[0]))
                            y_min = max(0, min(box1[1], box2[1]))
                            x_max = min(frame.shape[1], max(box1[2], box2[2]))
                            y_max = min(frame.shape[0], max(box1[3], box2[3]))
                            t = 10
                            crop = frame[y_min-t:y_max+t, x_min-t:x_max+t]

                            if crop.size == 0:
                                continue

                            input_tensor = transform(crop).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                features = feature_extractor(input_tensor)
                                output = classifier(features)
                                pred = (output.item() > CONFIDENCE_THRESHOLD)

                            if pred:
                                label = "Accident"
                                color = (0, 0, 255)
                                save_snapshot(crop, frame_id)

                                # Save snapshot with timestamped filename
                                snapshot_filename = f"accident_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                                snapshot_path = f"static/accidents/{snapshot_filename}"
                                cv2.imwrite(snapshot_path, crop)

                                # Update live accident data
                                with live_data_lock:
                                    live_data["accident_detection"] = [
                                        {
                                            "snapshot_url": f"/{snapshot_path}",
                                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "zone": "N/A",
                                            "confidence_score": np.round(float(output.item()),2),
                                            "prediction": "Accident",
                                            "confidence_level": "High" if output.item() > 0.8 else "Medium"
                                        }
                                    ]

                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                                cv2.putText(frame, label, (x_min, y_min - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


                # Draw all vehicles
                for box in vehicles:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

            

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.error("Failed to encode frame.")
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.exception(f"Exception in generate_frames: {e}")
                continue

        cap.release()


    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video source.")
            return

        while True:
            try:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.error("Could not read frame from video.")
                    break

                processed_frame, frame_data = process_frame(
                    frame=frame,
                    fps=int(fps),
                    colors=colors,
                    coordinates=coordinates,
                    view_transformers=view_transformers,
                    byte_tracker=byte_tracker,
                    selected_classes=selected_classes,
                    vehicle_model=VEHICLE_MODEL,
                    SOURCES=SOURCES,
                    TARGETS=TARGETS,
                    zone_annotators=zone_annotators,
                    box_annotators=box_annotators,
                    trace_annotators=trace_annotators,
                    line_zones=line_zones,
                    line_zone_annotators=line_zone_annotators,
                    label_annotators=label_annotators,
                    lines_start=lines_start,
                    lines_end=lines_end,
                    zones=zones
                )

                with live_data_lock:
                    live_data["traffic_stats"] = frame_data.get("traffic_stats", [])
                    live_data["alerts"] = frame_data.get("alerts", {})
                    live_data["accident_detection"] = frame_data.get("accident_detection", [])
                    live_data["analytics"] = frame_data.get("analytics", {})

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    logger.error("Failed to encode frame.")
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                time.sleep(max(1 / 25 - (time.time() - start_time), 0))

            except Exception as e:
                logger.exception(f"Exception in generate_frames: {e}")
                continue

        cap.release()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/camera-angles", response_class=HTMLResponse)
def camera_angles(request: Request):
    return templates.TemplateResponse("cameraAngles.html", {"request": request})

@app.get("/set-camera-mode/{mode}")
def set_camera_mode(mode: str):
    if mode not in ["live", "static","accident"]:
        raise HTTPException(status_code=400, detail="Mode must be 'live' or 'static'")
    with camera_mode_lock:
        camera_mode["mode"] = mode
    return {"message": f"Camera mode set to {mode}"}

@app.get("/video")
def video_feed():
    try:
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        return {"error": "Video feed error"}, 500

@app.get("/live-streaming", response_class=HTMLResponse)
def home(request: Request):
    with open("vehicle-tracking.txt", "r") as f:
        txt_data = f.readlines()
    return templates.TemplateResponse("tracking.html", {"request": request, "txt_data": txt_data})

@app.get("/log")
def get_log_data():
    try:
        with open("vehicle-tracking.txt", "r") as f:
            lines = f.readlines()
        return {"lines": lines}
    except FileNotFoundError:
        return {"lines": ["No vehicle data available."]}

@app.get("/accident-log")
def get_accident_data():
    try:
        with open("accident-log.txt", "r") as f:
            lines = f.readlines()
        return {"lines": lines[::-1]}
    except FileNotFoundError:
        return {"lines": ["No accident data available."]}

@app.get("/api/traffic-stats")
def get_traffic_stats():
    with live_data_lock:
        data = live_data.get("traffic_stats", [])
        logger.debug(f"/api/traffic-stats response: {data}")
    
    return {
        "zones": [
            {
                "zone_id": "Zone 1",
                "location": "West Direction 1",
                "total_vehicles": random.randint(50, 200),
                "density_level": random.choice(["Low", "Medium", "High"]),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vehicle_breakdown": {
                    "cars": random.randint(20, 100),
                    "buses": random.randint(5, 20),
                    "trucks": random.randint(10, 30),
                    "two_wheelers": random.randint(15, 50)
                },
                "average_speed": random.randint(30, 80)
            },
            {
                "zone_id": "Zone 2",
                "location": "East Direction 1",
                "total_vehicles": random.randint(50, 200),
                "density_level": random.choice(["Low", "Medium", "High"]),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vehicle_breakdown": {
                    "cars": random.randint(20, 100),
                    "buses": random.randint(5, 20),
                    "trucks": random.randint(10, 30),
                    "two_wheelers": random.randint(15, 50)
                },
                "average_speed": random.randint(30, 80)
            },
            {
                "zone_id": "Zone 3",
                "location": "East Direction 2",
                "total_vehicles": random.randint(50, 200),
                "density_level": random.choice(["Low", "Medium", "High"]),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vehicle_breakdown": {
                    "cars": random.randint(20, 100),
                    "buses": random.randint(5, 20),
                    "trucks": random.randint(10, 30),
                    "two_wheelers": random.randint(15, 50)
                },
                "average_speed": random.randint(30, 80)
            }
        ]
    }

@app.get("/api/alerts")
def get_alerts():
    with live_data_lock:
        alerts = live_data.get("alerts", {})
        logger.debug(f"/api/alerts response: {alerts}")
   
    return {
        "overspeeding": [
            {
                "vehicle_id": f"V{random.randint(1000, 9999)}",
                "speed": random.randint(80, 120),
                "zone": f"Zone {random.randint(1, 3)}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            } for _ in range(3)
        ],
        "stopped_vehicles": [
            {
                "vehicle_id": f"V{random.randint(1000, 9999)}",
                "duration": f"{random.randint(1, 10)} minutes",
                "zone": f"Zone {random.randint(1, 3)}"
            } for _ in range(2)
        ],
        "proximity_alerts": [
            {
                "vehicle1": f"V{random.randint(1000, 9999)}",
                "vehicle2": f"V{random.randint(1000, 9999)}",
                "distance": f"{random.randint(1, 5)} meters",
                "zone": f"Zone {random.randint(1, 3)}"
            } for _ in range(2)
        ],
        "accidents": [
            {
                "zone": f"Zone {random.randint(1, 3)}",
                "vehicles": [f"V{random.randint(1000, 9999)}" for _ in range(2)],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
    }

@app.get("/api/accident-detection")
def get_accident_detection():
    with live_data_lock:
        accident_data = live_data.get("accident_detection", [])

    if not accident_data:
        return {
            "accidents": [
                {
                    "snapshot_url": None,
                    "time": None,
                    "zone": None,
                    "confidence_score": 0.0,
                    "prediction": "None",
                    "confidence_level": "None"
                }
            ]
        }

    return {"accidents": accident_data}





@app.get("/api/analytics")
def get_analytics():
    with live_data_lock:
        analytics = live_data.get("analytics", {})
        logger.debug(f"/api/analytics response: {analytics}")
    
    return {
    "vehicle_count_over_time": {
        "labels": [f"{i}:00" for i in range(24)],
        "data": [random.randint(50, 200) for _ in range(24)]
    },
    "speed_distribution": {
        "ranges": ["0-20", "21-40", "41-60", "61-80", "81-100", "100+"],
        "counts": [random.randint(10, 50) for _ in range(6)]
    },
    "vehicle_type_distribution": {
        "labels": ["Cars", "Buses", "Trucks", "Two-wheelers"],
        "data": [
            [random.randint(20, 100) for _ in range(24)],
            [random.randint(5, 20) for _ in range(24)],
            [random.randint(10, 30) for _ in range(24)],
            [random.randint(15, 50) for _ in range(24)]
        ]
    },
    "zone_congestion": {
        "zones": ["Zone 1", "Zone 2", "Zone 3"],
        "congestion_levels": [random.randint(1, 100) for _ in range(3)]
    },
    "average_delay": {
        "labels": [f"{i}:00" for i in range(24)],
        "data": [random.randint(0, 30) for _ in range(24)]
    }
}

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8000")

def start_processing_loop():
    for _ in generate_frames():
        pass

if __name__ == "__main__":
    threading.Thread(target=start_processing_loop, daemon=True).start()
    threading.Thread(target=open_browser).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
