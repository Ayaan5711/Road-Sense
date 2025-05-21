
# Road-Sense: A Smart Traffic Monitoring System 🚦

## Overview

**Road-Sense** is a real-time intelligent traffic monitoring and accident detection system. It processes live or recorded video feeds to detect, track, and analyze vehicles using deep learning. The system includes features like speed estimation, vehicle classification, congestion detection, and accident alerts—presented via a FastAPI-powered live dashboard.

Built with:

* **YOLOv8** for vehicle and accident detection
* **OpenCV** and **Supervision** for video processing and annotation
* **FastAPI** for serving live video streams and insights

---

## 🔍 Key Features

### 🚗 Real-Time Vehicle Detection & Tracking

* Detects and tracks vehicles in traffic footage using YOLOv8
* Counts and traces vehicles across defined zones

### 🛑 Accident Detection

* Uses a separate YOLOv8 model to detect accidents
* Displays visual alerts on live stream

### 📏 Speed Estimation

* Estimates vehicle speed based on motion across zones
* Issues alerts when speed limits are exceeded

### 🛻 Vehicle Type Differentiation

* Differentiates between cars, bikes, buses, trucks
* Aggregates counts per vehicle type

### 🚥 Traffic Congestion Detection

* Monitors density and flow per lane/zone
* Detects and warns of potential congestion

### 🧊 Zone-Based Heatmaps *(optional enhancement)*

* Visualizes traffic patterns over time

### ⚠️ Proximity Alerts

* Warns when vehicles are dangerously close

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/road-sense.git
cd road-sense
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Add Models

Place your models in the `models/` directory:

* `yolov8s.pt` – for vehicle detection
* `best.pt` – your custom accident detection model

---

## 🧪 Running the Application

### A. Run the FastAPI Server with Auto-Browser Launch
```bash
python app.py
```
Your default browser will automatically open to:
```bash
http://127.0.0.1:8000/
```
### B. Manual Run (if auto-launch is not used)
```bash
uvicorn app:app --reload
```
Then open your browser and go to:
``` bash
http://127.0.0.1:8000/
```

* `/` – Landing page
* `/live-streaming` – Dashboard with vehicle logs
* `/video` – Live annotated video stream

---

## 🛠 Configuration

Update `app.py` or relevant config files to:

* Set your **video source** (YouTube link, IP camera, or file path)
* Adjust **zones** or **speed thresholds** as needed

---

## 🧾 Output

* **Live Annotated Video Feed**
  Displays:

  * Vehicle bounding boxes with IDs, types, and speeds
  * Accident detection alerts in real time

* **`vehicle-tracking.txt`**
  Logs vehicle detections, tracking IDs, zone transitions, and speed estimates.

* **`accident-log.txt`**
  Logs accident events detected by the model, including timestamps and zone locations.

---


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
* [Supervision by Roboflow](https://github.com/roboflow/supervision)
* [OpenCV](https://opencv.org/)
* [FastAPI](https://fastapi.tiangolo.com/)
* Special thanks to the traffic safety research community


