import numpy as np
import supervision as sv
import cv2

# def process_frame(frame: np.ndarray, fps, colors, coordinates, view_transformers, byte_tracker, selected_classes,
#                   vehicle_model, accident_model, SOURCES, TARGETS, zone_annotators, box_annotators,
#                   trace_annotators, line_zones, line_zone_annotators, label_annotators, lines_start,
#                   lines_end, zones) -> np.ndarray:
    
#     accident_model.model.names = {
#     0: 'Accident',
#     1: 'Accident',
#     2: 'Accident'  # Change label here
# }

#     speed_labels = [], [], []
#     zone_car_counts = [0] * len(zones)  # Initialize car counts for each zone
#     warning_message = ""
#     car_type_counts = [{} for _ in zones]  # Per-zone car type counts

#     results = vehicle_model(frame, imgsz=640, verbose=False)[0]

#     detections = sv.Detections.from_ultralytics(results)
#     detections = detections[np.isin(detections.class_id, selected_classes)]  # Filter on selected classes
#     detections = byte_tracker.update_with_detections(detections)

#     annotated_frame = frame.copy()
#     log_lines = []


#     # Run accident detection (new)
#     accident_results = accident_model(frame, imgsz=640, verbose=False)[0]
#     accident_detections = sv.Detections.from_ultralytics(accident_results)

#     # Skip frame if no detections
#     if len(detections) == 0 or detections.tracker_id is None or detections.tracker_id.shape[0] == 0:
#         height, width, _ = annotated_frame.shape
#         cv2.putText(annotated_frame, "Total Cars: 0", (width - 300, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         with open('vehicle-tracking.txt', 'a') as file:
#             file.write(f"\n--- Frame Data ---\nTotal Cars: 0\nNo vehicles detected in this frame.\n")
#         return annotated_frame

#     for i, (zone, zone_annotator, box_annotator, trace_annotator, line_zone, line_zone_annotator,
#             label_annotator, line_start, line_end, view_transformer, speed_label, coordinate) in enumerate(zip(
#             zones, zone_annotators, box_annotators, trace_annotators, line_zones, line_zone_annotators,
#             label_annotators, lines_start, lines_end, view_transformers, speed_labels, coordinates)):

#         direction_label = "Dir. West" if i == 0 else "Dir. East"
#         mask = zone.trigger(detections=detections)
#         detections_filtered = detections[mask]
#         zone_car_counts[i] = len(detections_filtered)

#         for class_id in detections_filtered.class_id:
#             class_name = vehicle_model.model.names[class_id]
#             car_type_counts[i][class_name] = car_type_counts[i].get(class_name, 0) + 1

#         points = detections_filtered.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
#         points = view_transformer.transform_points(points=points).astype(int)

#         for j, (point1, tracker_id1) in enumerate(zip(points, detections_filtered.tracker_id)):
#             for k, (point2, tracker_id2) in enumerate(zip(points, detections_filtered.tracker_id)):
#                 if j != k:
#                     distance = np.linalg.norm(point1 - point2)
#                     if distance < 20:
#                         warning_message += f"Cars {tracker_id1} and {tracker_id2} too close in {direction_label} (Zone {i+1}): {distance*0.025:.2f} meter\n"

#         for tracker_id, [_, y] in zip(detections_filtered.tracker_id, points):
#             coordinate[tracker_id].append(y)

#         for tracker_id in detections_filtered.tracker_id:
#             if len(coordinate[tracker_id]) < fps / 2:
#                 speed_label.append(f"#{tracker_id}")
#             else:
#                 try:
#                     coordinate_start = coordinate[tracker_id][-1]
#                     coordinate_end = coordinate[tracker_id][0]
#                     distance = abs(coordinate_start - coordinate_end)
#                     time = len(coordinate[tracker_id]) / fps
#                     speed = distance / time * 3.6
#                     speed_label.append(f"{int(speed)} km/h")
#                     log_lines.append(f"Car ID {tracker_id} Speed: {int(speed)} km/h (Zone {i+1}, {direction_label})")
#                 except Exception as e:
#                     speed_label.append(f"#{tracker_id}")
#                     print(f"An error occurred: {e}")

#         annotated_frame = zone_annotator.annotate(annotated_frame, f"{direction_label} : {line_zone.in_count if i == 0 else line_zone.out_count}")
#         annotated_frame = label_annotator.annotate(annotated_frame, detections_filtered, speed_label)
#         annotated_frame = box_annotator.annotate(annotated_frame, detections_filtered)
#         annotated_frame = trace_annotator.annotate(annotated_frame, detections_filtered)
#         annotated_frame = sv.draw_line(annotated_frame, line_start, line_end, colors.by_idx(i))

#         line_zone.trigger(detections=detections_filtered)

#     # Draw accident detections (red box)
#     for accident_det in accident_detections:
#         x1, y1, x2, y2 = map(int, accident_det[0][0:4])
#         confidence = f"{float(accident_det[2]):.2f}"
#         # confidence = accident_det[4] if len(accident_det) > 4 else 1.0
#         confidence = float(confidence)
#         if confidence > 0.70:
#             label = f"Accident {confidence}"
            
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#             with open('accident-log.txt', 'a') as accident_log:
#                 accident_log.write(f"Accident Detected - Coordinates: ({x1}, {y1}), ({x2}, {y2}), Confidence: {confidence:.2f}\n")



#     total_car_count = sum(zone_car_counts)
#     height, width, _ = annotated_frame.shape
#     cv2.putText(annotated_frame, f"Total Cars: {total_car_count}", (width - 300, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     with open('vehicle-tracking.txt', 'a') as file:
#         file.write(f"\n--- Frame Data ---\nTotal Cars: {total_car_count}\n")
#         for i, (count, type_count) in enumerate(zip(zone_car_counts, car_type_counts)):
#             direction = "Dir. West" if i == 0 else "Dir. East"
#             file.write(f"{direction} (Zone {i+1}): {count} cars\n  Types:\n")
#             for t, c in type_count.items():
#                 file.write(f"    {t}: {c}\n")
#             file.write(f"  Traffic Density (est.): {count} vehicles\n")
#         for line in log_lines:
#             file.write(line + "\n")
#         if warning_message:
#             file.write("WARNING: Cars too close!\n" + warning_message)

#     if warning_message:
#         cv2.putText(annotated_frame, "WARNING: Cars too close!", (width - 500, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     return annotated_frame


def process_frame(frame: np.ndarray, fps, colors, coordinates, view_transformers, byte_tracker, selected_classes,
                  vehicle_model, accident_model, SOURCES, TARGETS, zone_annotators, box_annotators,
                  trace_annotators, line_zones, line_zone_annotators, label_annotators, lines_start,
                  lines_end, zones) -> tuple:  # Return tuple now

    accident_model.model.names = {
        0: 'Accident',
        1: 'Accident',
        2: 'Accident'  # Change label here
    }

    speed_labels = [], [], []
    zone_car_counts = [0] * len(zones)  # Initialize car counts for each zone
    warning_message = ""
    car_type_counts = [{} for _ in zones]  # Per-zone car type counts

    results = vehicle_model(frame, imgsz=640, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_classes)]  # Filter on selected classes
    detections = byte_tracker.update_with_detections(detections)

    annotated_frame = frame.copy()
    log_lines = []

    # Run accident detection (new)
    accident_results = accident_model(frame, imgsz=640, verbose=False)[0]
    accident_detections = sv.Detections.from_ultralytics(accident_results)

    # Skip frame if no detections
    if len(detections) == 0 or detections.tracker_id is None or detections.tracker_id.shape[0] == 0:
        height, width, _ = annotated_frame.shape
        cv2.putText(annotated_frame, "Total Cars: 0", (width - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        with open('vehicle-tracking.txt', 'a') as file:
            file.write(f"\n--- Frame Data ---\nTotal Cars: 0\nNo vehicles detected in this frame.\n")

        # NEW: Return frame + empty live data
        frame_data = {
            "traffic_stats": [],
            "alerts": {"overspeeding": [], "proximity_alerts": [], "accidents": [], "stopped_vehicles": []},
            "accident_detection": [],
            "analytics": {"total_vehicles": 0, "warnings": "", "log_lines": []}
        }
        return annotated_frame, frame_data

    for i, (zone, zone_annotator, box_annotator, trace_annotator, line_zone, line_zone_annotator,
            label_annotator, line_start, line_end, view_transformer, speed_label, coordinate) in enumerate(zip(
            zones, zone_annotators, box_annotators, trace_annotators, line_zones, line_zone_annotators,
            label_annotators, lines_start, lines_end, view_transformers, speed_labels, coordinates)):

        direction_label = "Dir. West" if i == 0 else "Dir. East"
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        zone_car_counts[i] = len(detections_filtered)

        for class_id in detections_filtered.class_id:
            class_name = vehicle_model.model.names[class_id]
            car_type_counts[i][class_name] = car_type_counts[i].get(class_name, 0) + 1

        points = detections_filtered.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        for j, (point1, tracker_id1) in enumerate(zip(points, detections_filtered.tracker_id)):
            for k, (point2, tracker_id2) in enumerate(zip(points, detections_filtered.tracker_id)):
                if j != k:
                    distance = np.linalg.norm(point1 - point2)
                    if distance < 20:
                        warning_message += f"Cars {tracker_id1} and {tracker_id2} too close in {direction_label} (Zone {i+1}): {distance*0.025:.2f} meter\n"

        for tracker_id, [_, y] in zip(detections_filtered.tracker_id, points):
            coordinate[tracker_id].append(y)

        for tracker_id in detections_filtered.tracker_id:
            if len(coordinate[tracker_id]) < fps / 2:
                speed_label.append(f"#{tracker_id}")
            else:
                try:
                    coordinate_start = coordinate[tracker_id][-1]
                    coordinate_end = coordinate[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinate[tracker_id]) / fps
                    speed = distance / time * 3.6
                    speed_label.append(f"{int(speed)} km/h")
                    log_lines.append(f"Car ID {tracker_id} Speed: {int(speed)} km/h (Zone {i+1}, {direction_label})")
                except Exception as e:
                    speed_label.append(f"#{tracker_id}")
                    print(f"An error occurred: {e}")

        annotated_frame = zone_annotator.annotate(annotated_frame, f"{direction_label} : {line_zone.in_count if i == 0 else line_zone.out_count}")
        annotated_frame = label_annotator.annotate(annotated_frame, detections_filtered, speed_label)
        annotated_frame = box_annotator.annotate(annotated_frame, detections_filtered)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections_filtered)
        annotated_frame = sv.draw_line(annotated_frame, line_start, line_end, colors.by_idx(i))

        line_zone.trigger(detections=detections_filtered)

    # Draw accident detections (red box)
    accident_detection_results = []
    for accident_det in accident_detections:
        x1, y1, x2, y2 = map(int, accident_det[0][0:4])
        confidence = f"{float(accident_det[2]):.2f}"
        confidence = float(confidence)
        if confidence > 0.70:
            label = f"Accident {confidence}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            with open('accident-log.txt', 'a') as accident_log:
                accident_log.write(f"Accident Detected - Coordinates: ({x1}, {y1}), ({x2}, {y2}), Confidence: {confidence:.2f}\n")
            accident_detection_results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "label": "Accident"
            })

    total_car_count = sum(zone_car_counts)
    height, width, _ = annotated_frame.shape
    cv2.putText(annotated_frame, f"Total Cars: {total_car_count}", (width - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    with open('vehicle-tracking.txt', 'a') as file:
        file.write(f"\n--- Frame Data ---\nTotal Cars: {total_car_count}\n")
        for i, (count, type_count) in enumerate(zip(zone_car_counts, car_type_counts)):
            direction = "Dir. West" if i == 0 else "Dir. East"
            file.write(f"{direction} (Zone {i+1}): {count} cars\n  Types:\n")
            for t, c in type_count.items():
                file.write(f"    {t}: {c}\n")
            file.write(f"  Traffic Density (est.): {count} vehicles\n")
        for line in log_lines:
            file.write(line + "\n")
        if warning_message:
            file.write("WARNING: Cars too close!\n" + warning_message)

    if warning_message:
        cv2.putText(annotated_frame, "WARNING: Cars too close!", (width - 500, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # NEW: structured live data dict to return
    frame_data = {
        "traffic_stats": [
            {
                "zone": f"Zone {i+1}",
                "direction": "West" if i == 0 else "East",
                "vehicle_count": zone_car_counts[i],
                "vehicle_types": car_type_counts[i],
                "traffic_density_estimate": zone_car_counts[i]  # simple count as density estimate
            }
            for i in range(len(zones))
        ],
        "alerts": {
            "overspeeding": [],
            "proximity_alerts": warning_message.strip().split('\n') if warning_message else [],
            "accidents": accident_detection_results,
            "stopped_vehicles": []  #if needed
        },
        "accident_detection": accident_detection_results,
        "analytics": {
            "total_vehicles": total_car_count,
            "warnings": warning_message.strip(),
            "log_lines": log_lines
        }
    }

    return annotated_frame, frame_data
