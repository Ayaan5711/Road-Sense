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




app = FastAPI()

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

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(
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

        ret, buffer = cv2.imencode('.jpg', processed)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(max(1/25 - (time.time() - start_time), 0))
    
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
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

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

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

