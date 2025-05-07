import numpy as np
import supervision as sv



def create_zones(x1, y1, x2, y2, x3, y3, x4, y4, coef, video_info, colors):


    x1, y1, x2, y2, x3, y3, x4, y4 = map(
        lambda vals: list(map(lambda val: val * coef, vals)),
        [x1, y1, x2, y2, x3, y3, x4, y4]
    )

    # Find the centroid or third point from top of the polygon
    x14 = list(map(lambda x1, x4: (x1 + 2 * x4) / 3, x1, x4))
    y14 = list(map(lambda y1, y4: (y1 + 2 * y4) / 3, y1, y4))
    x23 = list(map(lambda x2, x3: (x2 + 2 * x3) / 3, x2, x3))
    y23 = list(map(lambda y2, y3: (y2 + 2 * y3) / 3, y2, y3))

    polygons = [np.array([[x1_, y1_], [x2_, y2_], [x3_, y3_], [x4_, y4_]], dtype=np.int32)
                for x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_
                in zip(x1, y1, x2, y2, x3, y3, x4, y4)]
    
    zones = [sv.PolygonZone(p, video_info.resolution_wh) for p in polygons]



    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=2,
            text_thickness=1,
            text_scale=0.5,
        )
        for index, zone
        in enumerate(zones)
    ]
    label_annotators = [
        sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            color=colors.by_idx(index),
            text_thickness=1,
            text_scale=0.5,
        )
        for index
        in range(len(zones))
    ]

    box_annotators = [
        sv.BoundingBoxAnnotator(
            color=colors.by_idx(index),
            thickness=1,
            )
        for index
        in range(len(polygons))
    ]

    trace_annotators = [
        sv.TraceAnnotator(
            color=colors.by_idx(index),
            thickness=1,
            trace_length=video_info.fps * 1.5,
            position=sv.Position.BOTTOM_CENTER,
            )
        for index
        in range(len(polygons))
    ]
    lines_start = [
        sv.Point(x14, y14)
        for x14, y14
        in zip(x14, y14)
    ]
    lines_end = [
        sv.Point(x23, y23)
        for x23, y23
        in zip(x23, y23)
    ]
    positions = [
        (sv.Position.CENTER, sv.Position.CENTER),
        (sv.Position.CENTER, sv.Position.CENTER),
        (sv.Position.CENTER, sv.Position.CENTER),
    ]
    line_zones = [
        sv.LineZone(start=line_start, end=line_end,
                    triggering_anchors=position)
        for line_start, line_end, position
        in zip(lines_start, lines_end, positions)
    ]
    # for automatic line zone annotator not use here want to use a custom one
    line_zone_annotators = [
        sv.LineZoneAnnotator(thickness=1,
                                color=colors.by_idx(index),
                                text_thickness=1,
                                text_scale=0.5,
                                text_offset=4)
        for index
        in range(len(line_zones))
    ]

    SOURCES = np.array([[
        [x4[0], y4[0]],
        [x3[0], y3[0]],
        [x2[0], y2[0]],
        [x1[0], y1[0]]

    ], [[x4[1], y4[1]],
        [x3[1], y3[1]],
        [x2[1], y2[1]],
        [x1[1], y1[1]]],
            [[x4[2], y4[2]],
            [x3[2], y3[2]],
            [x2[2], y2[2]],
            [x1[2], y1[2]]
            ]
            ])
    
    # zone1 in meters
    TARGET_WIDTH = 6
    TARGET_HEIGHT = 75
    TARGETS = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ])
    # zone 2 in meters
    TARGET_WIDTH = 6
    TARGET_HEIGHT = 85
    TARGETS = np.append(TARGETS, np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]), axis=0)
    # zone3 in meters
    TARGET_WIDTH = 6
    TARGET_HEIGHT = 80
    TARGETS = np.append(TARGETS, np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]), axis=0)

    TARGETS = TARGETS.reshape(3, 4, 2)

    return [SOURCES, TARGETS, zone_annotators,
                    box_annotators,
                    trace_annotators,
                    line_zones,
                    line_zone_annotators,
                    label_annotators,
                    lines_start,
                    lines_end,zones, x1, y1, x2, y2, x3, y3, x4, y4]
