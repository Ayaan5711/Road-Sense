# Road-Sense: A Smart Traffic Monitoring System

Traffic Monitoring and Speed Estimation System
This project is a real-time traffic monitoring and speed estimation system designed to process video input from traffic cameras. The system uses a combination of YOLO for object detection, OpenVINO for accelerated inference, and custom logic for speed estimation, vehicle type differentiation, and congestion detection. The system detects and tracks vehicles, monitors their speeds, and provides various features like vehicle type differentiation, traffic congestion detection, zone-based vehicle heatmaps, and alerts for vehicles that are too close.

Features
Real-time Object Detection and Tracking:

Detects and tracks vehicles (car, motorcycle, bus, truck) from live traffic camera feeds.
Provides a real-time count of vehicles in each lane.
Speed Estimation:

Calculates and displays the speed of each detected vehicle.
Alerts when vehicles exceed the speed limit.
Vehicle Type Differentiation:

Differentiates between various types of vehicles and displays counts by vehicle type.
Traffic Congestion Detection:

Monitors traffic density in each zone and detects potential congestion.
Displays alerts when congestion is detected.
Zone-Based Vehicle Heatmaps:

Provides heatmaps of vehicle activity in different zones.
Allows for zone-specific analysis of traffic patterns.
Proximity Alerts:

Detects when vehicles are too close to each other and provides alerts on the screen and in the backend with distance and zone information.