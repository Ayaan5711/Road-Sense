# Road-Sense: A Smart Traffic Monitoring System

## Overview

**Road-Sense** is a real-time traffic monitoring and speed estimation system designed to process video input from traffic cameras. The system utilizes advanced technologies to provide comprehensive traffic analysis, including object detection, speed estimation, vehicle type differentiation, and congestion detection. Built using YOLO for object detection, OpenVINO for accelerated inference, and custom logic for various traffic monitoring features, Road-Sense enhances traffic management and safety.

## Features

### Real-time Object Detection and Tracking
- Detects and tracks vehicles (car, motorcycle, bus, truck) from live traffic camera feeds.
- Provides a real-time count of vehicles in each lane.

### Speed Estimation
- Calculates and displays the speed of each detected vehicle.
- Alerts when vehicles exceed the speed limit.

### Vehicle Type Differentiation
- Differentiates between various types of vehicles.
- Displays counts by vehicle type for detailed traffic analysis.

### Traffic Congestion Detection
- Monitors traffic density in each zone.
- Detects and alerts when congestion is identified.

### Zone-Based Vehicle Heatmaps
- Generates heatmaps to visualize vehicle activity across different zones.
- Facilitates zone-specific traffic pattern analysis.

### Proximity Alerts
- Detects when vehicles are too close to each other.
- Provides on-screen and backend alerts with distance and zone information.

## Getting Started

To get started with Road-Sense, clone the repository and follow the instructions below:

### Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/road-sense.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd road-sense
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

- Ensure you have a traffic camera feed URL or video file.
- Update the configuration settings in `config.py` as needed.

### Running the System

1. **Run the Script:**

    ```bash
    python main.py
    ```

2. **Monitor Output:**

    - The system will display real-time video with annotated vehicle information.
    - Alerts and statistics will be shown on the video feed.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- YOLO for object detection
- OpenVINO for accelerated inference
- OpenCV for image processing

For more details and documentation, please visit the [project repository](https://github.com/yourusername/road-sense).



## Note from Zaib
In order to run FastAPI server run below code:
1. 
    ```bash
    pip install -r requirements.txt
    ```
2. 
    ```bash
    uvicorn app:app --reload
    ```

3. Naviagte to ```http://127.0.0.1:8000/video```
