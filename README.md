# 🚦 RoadSense: Intelligent Traffic Monitoring & Accident Detection System

RoadSense is an AI-powered, real-time vehicle tracking, traffic analytics, and accident detection system. It leverages advanced computer vision techniques including YOLOv8 for vehicle detection, ByteTrack for tracking, and a custom DenseNet201 + ANN pipeline for accident classification. A FastAPI backend serves an interactive dashboard with live video streaming and real-time alerts.

## 📷 Key Features

* 🚘 Vehicle Detection using YOLOv8
* 📍 Zone-based Traffic Monitoring
* 📊 Traffic Density & Vehicle Type Statistics
* 🚦 Real-time Speed Estimation via Perspective Transform
* ⚠️ Accident Detection using YOLOv8 + AABB + DenseNet201 + ANN Classifier
* 🔊 Alerts for Overspeeding, Proximity, Stopped Vehicles & Accidents
* 📝 Per-frame Logging to vehicle-tracking.txt and accident-log.txt
* 🖥️ FastAPI Dashboard for Live Analytics and Stream Switching (live/static/accident)

## 📁 Project Structure

```bash
RoadSense/
├── app.py                    # FastAPI backend with endpoints
├── models/
│   ├── yolov8s.pt            # Vehicle Detection Model
│   └── model_epoch_50.pth    # Accident Detection Model 
├── src/
|   ├── processing.py         # Core frame processing logic (detection, tracking, logging)
│   ├── accident_detection.py # Modular accident detection pipeline
│   ├── tracking_utils.py     # ByteTrack integration
│   ├── video_utils.py        # YouTube/local stream handler
│   ├── zone_utils.py         # Polygonal zone management
│   ├── transform_utils.py    # Perspective transform & speed conversion
├── static/
│   ├── css/                  # Styling for dashboard
│   └── js/                   # Optional: frontend scripts
├── templates/
│   ├── landing.html          # Welcome page
│   ├── cameraAngles.html     # Choose camera mode
│   └── tracking.html         # Live video + analytics display
├── videos/
│   ├── recorded_stream_large.mp4
│   └── accident_t2.mp4
├── vehicle-tracking.txt     # Per-frame traffic log
├── accident-log.txt         # Confirmed accidents log
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🚀 How It Works

1. 🎥 Capture video from YouTube or local source using vidgear/OpenCV.
2. 🧠 Run YOLOv8 detection on each frame.
3. 🎯 Track vehicles using ByteTrack.
4. 🚗 Estimate speed using vertical position history + perspective transform.
5. 🛑 Detect accidents using a pipeline:

    AABB overlap → Crop pair → DenseNet201 feature extract → ANN prediction
6. 💡 Overlay annotations, log results, and stream to FastAPI frontend.


## 🖥️ System Architecture

The application follows a modular pipeline architecture:

1.  **Video Input:** The system captures frames from a video source (local file or live stream) using OpenCV.
2.  **Detection & Tracking:** Each frame is passed to the YOLOv8 model for vehicle detection. The resulting detections are then fed into the ByteTrack algorithm to track objects across frames.
3.  **Data Processing (`processing.py`):** Custom logic is applied to the tracked objects to:
    -   Check which zone they are in.
    -   Calculate their speed.
    -   Check for proximity to other vehicles.
    -   Count vehicles per zone and type.
4.  **Accident Detection:** A separate pipeline runs for the accident-focused video mode, using a specialized model to identify crash events.
5.  **FastAPI Backend (`app.py`):**
    -   Serves the processed and annotated video stream.
    -   Provides REST API endpoints for the frontend to fetch live data (traffic stats, alerts, analytics).
6.  **Web Dashboard (UI):** A web browser renders the `tracking.html` template, which displays the video feed and uses JavaScript to periodically call the API endpoints to refresh the data dashboards.


## 🧠 Accident Detection Pipeline

* Detection: YOLOv8 bounding boxes
* Collision Check: Axis-Aligned Bounding Box overlap
* Feature Extraction: DenseNet201 (pretrained)
* Classification: ANN model with confidence threshold (e.g., 0.82)
* Result: Snapshot saved, accident-log.txt updated, alert sent to UI

## 🖥️ FastAPI Endpoints

* / : Landing page
* /camera-angles : Choose mode (live / static / accident)
* /set-camera-mode/{mode}
* /video : MJPEG stream
* /live-streaming : Dashboard with vehicle logs
* /log : JSON log of all vehicle tracking data
* /accident-log : JSON log of accidents
* /api/traffic-stats : Per-zone stats
* /api/alerts : Real-time alerts
* /api/analytics : Graph analytics (counts, speed, type dist.)
* /api/accident-detection : List of accident snapshot events

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Ayaan5711/Road-Sense.git
cd Road-Sense
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download YOLOv8 and DenseNet201 weights and place in appropriate folders.

4. Run the app:

```bash
python app.py
```

Then open: [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 📦 Requirements

* Python 3.8+
* OpenCV
* Ultralytics YOLO
* Supervision
* Torch + torchvision
* FastAPI
* Uvicorn
* vidgear
* scikit-learn, numpy, pandas

## 📸 Snapshots

All detected accidents save snapshots with cropped vehicles. These are useful for audits, retraining, and reports.

## 📈 Example Use Cases

* Smart Traffic Monitoring by City Authorities
* Accident Detection for Surveillance
* Traffic Pattern Analytics for Urban Planning
* Research in Intelligent Transport Systems (ITS)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
* [Supervision by Roboflow](https://github.com/roboflow/supervision)
* [OpenCV](https://opencv.org/)
* [FastAPI](https://fastapi.tiangolo.com/)
* Special thanks to the traffic safety research community


## 📬 Contact

For questions or collaborations, please reach out to:

* Team Decodians

