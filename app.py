import webbrowser
import uvicorn
import threading
import cv2
import time
import numpy as np
from collections import defaultdict, deque
from src.video_utils import video_manifest_extractor
from src.zone_utils import create_zones
from src.tracking_utils import initialize_tracking
from src.processing import process_frame
from src.transform_utils import ViewTransformer
from ultralytics import YOLO
import supervision as sv

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from datetime import datetime
import random
from typing import List, Dict
import json
from threading import Lock

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

# Lock for thread-safe updates
live_data_lock = Lock()


app = FastAPI()

# Mount static folder for CSS and other static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
SOURCE = "https://youtu.be/z545k7Tcb5o"
VIDEO = video_manifest_extractor(SOURCE)
VEHICLE_MODEL_PATH = "models/yolov8s.pt"
ACCIDENT_MODEL_PATH = "models/best.pt"
VEHICLE_MODEL = YOLO(VEHICLE_MODEL_PATH)
ACCIDENT_MODEL = YOLO(ACCIDENT_MODEL_PATH)

colors = sv.ColorPalette.LEGACY
video_info = sv.VideoInfo.from_video_path(VIDEO)
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

def generate_frames():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    while True:
        try:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Process frame and get both the visual and data outputs
            processed_frame, frame_data = process_frame(
                frame=frame,
                fps=int(fps),
                colors=colors,
                coordinates=coordinates,
                view_transformers=view_transformers,
                byte_tracker=byte_tracker,
                selected_classes=selected_classes,
                vehicle_model=VEHICLE_MODEL,
                accident_model=ACCIDENT_MODEL,
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

            # ✅ Update the live_data dictionary
            with live_data_lock:
                live_data["traffic_stats"] = frame_data.get("traffic_stats", [])
                live_data["alerts"] = frame_data.get("alerts", {})
                live_data["accident_detection"] = frame_data.get("accident_detection", [])
                live_data["analytics"] = frame_data.get("analytics", {})

            # Encode the processed frame for video stream
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                print("Error: Could not encode frame")
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(max(1 / 25 - (time.time() - start_time), 0))

        except Exception as e:
            print(f"Error in frame generation: {str(e)}")
            continue

    cap.release()



templates = Jinja2Templates(directory="templates")  # Ensure you create this directory

@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/camera-angles", response_class=HTMLResponse)
def camera_angles(request: Request):
    return templates.TemplateResponse("cameraAngles.html", {"request": request})

@app.get("/live-streaming", response_class=HTMLResponse)
def home(request: Request):
    # Read the content of the txt file
    with open("vehicle-tracking.txt", "r") as f:
        txt_data = f.readlines()
    return templates.TemplateResponse("tracking.html", {"request": request, "txt_data": txt_data})


@app.get("/video")
def video_feed():
    try:
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        print(f"Error in video feed: {str(e)}")
        return {"error": "Video feed error"}, 500


@app.get("/log")
def get_log_data():
    try:
        with open("vehicle-tracking.txt", "r") as f:
            lines = f.readlines()
        # Return lines in chronological order (oldest first)
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

    if data:
        return {"zones": data}

    # Fallback  data
    return {
        "zones": [
            {
                "zone_id": "Zone 1",
                "location": "Main Intersection",
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
                "location": "Highway Entry",
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

    if alerts:
        return alerts

    # Fallback  alerts
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
    # Mock data for accident detection
    return {
        "accidents": [
            {
                "snapshot_url": "/static/accident_snapshot.jpg",  # This would be a real image in production
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "zone": f"Zone {random.randint(1, 3)}",
                "confidence_score": round(random.uniform(0.7, 0.95), 2),
                "prediction": "Accident",
                "confidence_level": f"{random.randint(70, 95)}%"
            } for _ in range(1)
        ]
    }

@app.get("/api/analytics")
def get_analytics():
    with live_data_lock:
        analytics = live_data.get("analytics", {})

    if analytics:
        return analytics

    # Fallback  analytics
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

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
