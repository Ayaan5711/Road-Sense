from src.video_utils import extract_video_url, video_manifest_extractor
from src.zone_utils import create_zones
from src.tracking_utils import initialize_tracking
from src.processing import process_frame
import cv2
import time
import supervision as sv
from ultralytics import YOLO
from src.transform_utils import ViewTransformer
import numpy as np
from collections import defaultdict, deque



def main():

    # 1
    SOURCE = "https://youtu.be/z545k7Tcb5o"
    VIDEO = video_manifest_extractor(SOURCE)

    # 2
    PATH = "models/yolov8s.pt"
    MODEL = YOLO(PATH)
    CLASS_NAMES_DICT = MODEL.model.names

    # 3
    colors = sv.ColorPalette.LEGACY
    video_info = sv.VideoInfo.from_video_path(VIDEO)
    coef = video_info.width / 1280
    # Coordinates
    x1 = [-160, -25, 971]; y1 = [405, 710, 671]
    x2 = [112, 568, 1480]; y2 = [503, 710, 671]
    x3 = [557, 706, 874];  y3 = [195, 212, 212]
    x4 = [411, 569, 749];  y4 = [195, 212, 212]
    SOURCES, TARGETS, zone_annotators,box_annotators,trace_annotators,line_zones,line_zone_annotators,label_annotators,lines_start,lines_end,zones, x1, y1, x2, y2, x3, y3, x4, y4 = create_zones(x1, y1, x2, y2, x3, y3, x4, y4, coef, video_info, colors)
    
    # 4
    byte_tracker, fps_monitor = initialize_tracking(video_info)

    # 5
    view_transformers = [
        ViewTransformer(source=s, target=t)
        for s, t
        in zip(SOURCES, TARGETS)
    ]


    # 6
    selected_classes = [2, 3, 5, 7] 
    # initialize the dictionary that we will use to store the coordinates for each zone
    coordinates = defaultdict(lambda: deque(maxlen=30))
    coordinates = defaultdict(lambda: deque(maxlen=30))
    coordinates = np.append(coordinates, defaultdict(lambda: deque(maxlen=30)))
    coordinates = np.append(coordinates, defaultdict(lambda: deque(maxlen=30)))

    # 7
    cap = cv2.VideoCapture(VIDEO)
    fps = video_info.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"FPS: {fps}")
    print(f"image : {width}x{height}")

    # 8
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        show = process_frame(frame, int(fps),colors,coordinates,view_transformers,byte_tracker,selected_classes,MODEL,SOURCES, TARGETS, zone_annotators,box_annotators,trace_annotators,line_zones,line_zone_annotators,label_annotators,lines_start,lines_end,zones) # Supply other annotators/objects
        fps_monitor.tick()
        cv2.putText(show, f"FPS: {fps:.0f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Counting - Speed Estimation", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(max(1/25 - (time.time() - start_time), 0))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
