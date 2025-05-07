

import numpy as np
import supervision as sv
import cv2


# Define processing function
def process_frame(frame: np.ndarray, fps, colors, coordinates, view_transformers, byte_tracker, selected_classes,
                  model, SOURCES, TARGETS, zone_annotators, box_annotators, trace_annotators, line_zones,
                  line_zone_annotators, label_annotators, lines_start, lines_end, zones) -> np.ndarray:
    
    speed_labels = [], [], []
    zone_car_counts = [0] * len(zones)  # Initialize car counts for each zone
    warning_message = ""
    car_type_counts = [{} for _ in zones]  # Per-zone car type counts

    results = model(frame, imgsz=640, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_classes)]  # Filter on selected classes
    detections = byte_tracker.update_with_detections(detections)

    annotated_frame = frame.copy()
    log_lines = []

    # === Skip processing if no detections ===
    if len(detections) == 0 or detections.tracker_id is None or detections.tracker_id.shape[0] == 0:
        total_car_count = 0
        height, width, _ = annotated_frame.shape
        text_position = (width - 300, 50)

        cv2.putText(annotated_frame, "Total Cars: 0", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        with open('vehicle-tracking.txt', 'a') as file:
            file.write(f"\n--- Frame Data ---\n")
            file.write(f"Total Cars: 0\n")
            file.write("No vehicles detected in this frame.\n")

        return annotated_frame

    # === Proceed with processing ===
    for i, (zone,
            zone_annotator,
            box_annotator,
            trace_annotator,
            line_zone,
            line_zone_annotator,
            label_annotator,
            line_start,
            line_end,
            view_transformer,
            speed_label,
            coordinate) in enumerate(zip(
                zones,
                zone_annotators,
                box_annotators,
                trace_annotators,
                line_zones,
                line_zone_annotators,
                label_annotators,
                lines_start,
                lines_end,
                view_transformers,
                speed_labels,
                coordinates)):

        direction_label = "Dir. West" if i == 0 else "Dir. East"
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        zone_car_counts[i] = len(detections_filtered)

        for class_id in detections_filtered.class_id:
            class_name = model.model.names[class_id]
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

        annotated_frame = zone_annotator.annotate(
            scene=annotated_frame,
            label=f"{direction_label} : {line_zone.in_count if i == 0 else line_zone.out_count}")

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections_filtered,
            labels=speed_label)

        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections_filtered)

        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections_filtered)

        annotated_frame = sv.draw_line(
            scene=annotated_frame,
            start=line_start,
            end=line_end,
            color=colors.by_idx(i))

        line_zone.trigger(detections=detections_filtered)

    # Total and per-lane statistics
    total_car_count = sum(zone_car_counts)
    height, width, _ = annotated_frame.shape
    text_position = (width - 300, 50)
    warning_text_position = (width - 500, 100)

    cv2.putText(annotated_frame, f"Total Cars: {total_car_count}", text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Log all collected data
    with open('vehicle-tracking.txt', 'a') as file:
        file.write(f"\n--- Frame Data ---\n")
        file.write(f"Total Cars: {total_car_count}\n")

        for i, (count, type_count) in enumerate(zip(zone_car_counts, car_type_counts)):
            direction = "Dir. West" if i == 0 else "Dir. East"
            file.write(f"{direction} (Zone {i+1}): {count} cars\n")
            file.write("  Types:\n")
            for t, c in type_count.items():
                file.write(f"    {t}: {c}\n")
            file.write(f"  Traffic Density (est.): {count} vehicles\n")

        for line in log_lines:
            file.write(line + "\n")

        if warning_message:
            file.write("WARNING: Cars too close!\n")
            file.write(warning_message)

    if warning_message:
        cv2.putText(annotated_frame, "WARNING: Cars too close!", warning_text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return annotated_frame
