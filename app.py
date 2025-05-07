import cv2
import time
import numpy as np
from collections import defaultdict, deque
from src.video_utils import extract_video_url, video_manifest_extractor
from src.zone_utils import create_zones
from src.tracking_utils import initialize_tracking
from src.processing import process_frame
from src.transform_utils import ViewTransformer
from ultralytics import YOLO
import supervision as sv

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Global variables
SOURCE = "https://youtu.be/z545k7Tcb5o"
VIDEO = video_manifest_extractor(SOURCE)
PATH = "models/yolov8s.pt"
MODEL = YOLO(PATH)
CLASS_NAMES_DICT = MODEL.model.names
colors = sv.ColorPalette.LEGACY
video_info = sv.VideoInfo.from_video_path(VIDEO)
coef = video_info.width / 1280

x1 = [-160, -25, 971]; y1 = [405, 710, 671]
x2 = [112, 568, 1480]; y2 = [503, 710, 671]
x3 = [557, 706, 874];  y3 = [195, 212, 212]
x4 = [411, 569, 749];  y4 = [195, 212, 212]
SOURCES, TARGETS, zone_annotators, box_annotators, trace_annotators, line_zones, line_zone_annotators, label_annotators, lines_start, lines_end, zones, *_ = create_zones(x1, y1, x2, y2, x3, y3, x4, y4, coef, video_info, colors)

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
            frame, int(fps), colors, coordinates, view_transformers, byte_tracker,
            selected_classes, MODEL, SOURCES, TARGETS,
            zone_annotators, box_annotators, trace_annotators,
            line_zones, line_zone_annotators, label_annotators,
            lines_start, lines_end, zones
        )

        ret, buffer = cv2.imencode('.jpg', processed)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(max(1/25 - (time.time() - start_time), 0))
    
    cap.release()




templates = Jinja2Templates(directory="templates")  # Ensure you create this directory

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")



