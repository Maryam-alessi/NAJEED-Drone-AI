import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

KEYPOINT_CONFIDENCE = 0.35

# Wound segmentation model
WOUND_CONFIDENCE = 0.70
WOUND_IOU = 0.50

# A wound must mostly belong to one detected person
MIN_WOUND_INSIDE_PERSON_RATIO = 0.50

# Reject extremely large masks compared with the person's box
MAX_WOUND_TO_PERSON_AREA_RATIO = 0.25

# Body-part localization
MAX_SKELETON_DISTANCE = 55

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# BODY PART LOCALIZATION + WOUND MASK HELPERS
# =========================================================
def valid_kp(xy, conf, idx):
    return (
        conf[idx] >= KEYPOINT_CONFIDENCE
        and not (
            xy[idx][0] == 0
            and xy[idx][1] == 0
        )
    )

def point_distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )
    )

def point_to_segment_distance(point, start, end):
    p = np.asarray(point, dtype=float)
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)

    segment = e - s
    length_squared = np.dot(segment, segment)

    if length_squared == 0:
        return float(np.linalg.norm(p - s))

    t = np.clip(
        np.dot(p - s, segment) / length_squared,
        0.0,
        1.0
    )

    closest = s + t * segment

    return float(np.linalg.norm(p - closest))

def min_segment_distance(point, xy, conf, segments):
    best = float("inf")

    for start_idx, end_idx in segments:

        if (
            valid_kp(xy, conf, start_idx)
            and valid_kp(xy, conf, end_idx)
        ):
            d = point_to_segment_distance(
                point,
                xy[start_idx],
                xy[end_idx]
            )

            best = min(best, d)

    return best

def localize_body_part(wound_point, xy, conf):

    candidates = {
        "LEFT ARM": min_segment_distance(
            wound_point,
            xy,
            conf,
            [(5, 7), (7, 9)]
        ),

        "RIGHT ARM": min_segment_distance(
            wound_point,
            xy,
            conf,
            [(6, 8), (8, 10)]
        ),

        "LEFT LEG": min_segment_distance(
            wound_point,
            xy,
            conf,
            [(11, 13), (13, 15)]
        ),

        "RIGHT LEG": min_segment_distance(
            wound_point,
            xy,
            conf,
            [(12, 14), (14, 16)]
        ),

        "TORSO": min_segment_distance(
            wound_point,
            xy,
            conf,
            [(5, 6), (5, 11), (6, 12), (11, 12)]
        ),
    }

    head_points = []

    for idx in (0, 1, 2, 3, 4):
        if valid_kp(xy, conf, idx):
            head_points.append(xy[idx])

    if head_points:
        candidates["HEAD"] = min(
            point_distance(wound_point, p)
            for p in head_points
        )
    else:
        candidates["HEAD"] = float("inf")

    best_part = min(
        candidates,
        key=candidates.get
    )

    if candidates[best_part] > MAX_SKELETON_DISTANCE:
        return None

    return best_part

def polygon_to_binary_mask(polygon, width, height):
    """
    Convert Ultralytics segmentation polygon to a binary image mask.
    """
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if polygon is None or len(polygon) < 3:
        return mask

    pts = np.asarray(
        polygon,
        dtype=np.int32
    )

    pts[:, 0] = np.clip(
        pts[:, 0],
        0,
        width - 1
    )

    pts[:, 1] = np.clip(
        pts[:, 1],
        0,
        height - 1
    )

    cv2.fillPoly(
        mask,
        [pts],
        255
    )

    return mask

def person_mask_overlap(mask, person_box):
    """
    Returns:
    - fraction of wound mask inside person's box
    - wound area / person box area
    - center of wound pixels that are actually inside the person box
    """
    x1, y1, x2, y2 = map(
        int,
        person_box
    )

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(FRAME_WIDTH, x2)
    y2 = min(FRAME_HEIGHT, y2)

    wound_area = int(
        np.count_nonzero(mask)
    )

    if wound_area == 0:
        return 0.0, 0.0, None

    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, None

    person_area = (
        (x2 - x1)
        * (y2 - y1)
    )

    if person_area <= 0:
        return 0.0, 0.0, None

    inside = np.zeros_like(mask)

    inside[
        y1:y2,
        x1:x2
    ] = mask[
        y1:y2,
        x1:x2
    ]

    inside_area = int(
        np.count_nonzero(inside)
    )

    inside_ratio = (
        inside_area
        / wound_area
    )

    wound_to_person_ratio = (
        inside_area
        / person_area
    )

    ys, xs = np.where(
        inside > 0
    )

    if len(xs) == 0:
        center = None
    else:
        center = (
            int(np.mean(xs)),
            int(np.mean(ys))
        )

    return (
        inside_ratio,
        wound_to_person_ratio,
        center
    )

def assign_wound_to_person(mask, people):
    """
    Associate the wound mask with the person that contains
    the largest proportion of the wound.
    """
    best_person = None
    best_inside_ratio = 0.0
    best_area_ratio = 0.0
    best_center = None

    for person in people:

        (
            inside_ratio,
            area_ratio,
            center
        ) = person_mask_overlap(
            mask,
            person["box"]
        )

        if inside_ratio > best_inside_ratio:

            best_inside_ratio = inside_ratio
            best_area_ratio = area_ratio
            best_person = person
            best_center = center

    if best_person is None:
        return None

    if (
        best_inside_ratio
        < MIN_WOUND_INSIDE_PERSON_RATIO
    ):
        return None

    if (
        best_area_ratio
        > MAX_WOUND_TO_PERSON_AREA_RATIO
    ):
        return None

    if best_center is None:
        return None

    return {
        "person": best_person,
        "center": best_center,
        "inside_ratio": best_inside_ratio,
        "area_ratio": best_area_ratio,
    }

def draw_wound_mask_overlay(
    display_frame,
    binary_mask
):
    """
    Paint accepted segmentation masks transparently in red.
    """
    overlay = display_frame.copy()

    overlay[
        binary_mask > 0
    ] = (0, 0, 255)

    cv2.addWeighted(
        overlay,
        0.35,
        display_frame,
        0.65,
        0,
        display_frame
    )

# =========================================================
# LOCAL WOUND MODEL LOADING
# =========================================================
def find_local_model(possible_names):
    """
    Find a model in the same folder as this script.
    This prevents Ultralytics from trying to download a missing model
    while the laptop is connected to the Tello Wi-Fi.
    """
    for name in possible_names:
        path = BASE_DIR / name
        if path.exists():
            return str(path)

    names = ", ".join(possible_names)

    raise FileNotFoundError(
        f"Model not found. Put one of these files in the same folder "
        f"as this script: {names}"
    )


def load_wound_model():
    wound_model_path = find_local_model(
        [
            "best.pt"
        ]
    )

    print(
        f"Loading Wound model: "
        f"{Path(wound_model_path).name}"
    )
    wound_model = YOLO(
        wound_model_path
    )

    print(
        "Wound model classes:",
        wound_model.names
    )

    return wound_model

# =========================================================
# WOUND DETECTION
# =========================================================
def process_wounds(
    frame,
    display_frame,
    people,
    wound_model
):
    wound_results = (
        wound_model.predict(
            source=frame,
            conf=WOUND_CONFIDENCE,
            iou=WOUND_IOU,
            verbose=False
        )
    )

    wound_result = (
        wound_results[0]
    )

    accepted_wounds = 0

    if (
        len(people) > 0
        and
        wound_result.boxes
        is not None
        and
        wound_result.masks
        is not None
    ):

        wound_boxes = (
            wound_result
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        wound_scores = (
            wound_result
            .boxes
            .conf
            .cpu()
            .numpy()
        )

        wound_classes = (
            wound_result
            .boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )

        polygons = (
            wound_result
            .masks
            .xy
        )

        wound_count = min(
            len(wound_boxes),
            len(polygons)
        )

        for wound_idx in range(
            wound_count
        ):

            polygon = (
                polygons[
                    wound_idx
                ]
            )

            binary_mask = (
                polygon_to_binary_mask(
                    polygon,
                    FRAME_WIDTH,
                    FRAME_HEIGHT
                )
            )

            assignment = (
                assign_wound_to_person(
                    binary_mask,
                    people
                )
            )

            # Ignore a prediction that does not really
            # belong to any detected person.
            if assignment is None:
                continue

            person = (
                assignment["person"]
            )

            wound_center = (
                assignment["center"]
            )

            body_part = (
                localize_body_part(
                    wound_center,
                    person["xy"],
                    person["conf"]
                )
            )

            # If it is too far from the skeleton,
            # do not force a wrong body location.
            if body_part is None:
                continue

            accepted_wounds += 1

            # Draw the actual segmentation mask
            draw_wound_mask_overlay(
                display_frame,
                binary_mask
            )

            wx1, wy1, wx2, wy2 = (
                wound_boxes[
                    wound_idx
                ].astype(int)
            )

            wound_conf = float(
                wound_scores[
                    wound_idx
                ]
            )

            class_id = int(
                wound_classes[
                    wound_idx
                ]
            )

            model_class = (
                wound_model.names[
                    class_id
                ]
                if isinstance(
                    wound_model.names,
                    dict
                )
                else
                str(class_id)
            )

            cx, cy = wound_center

            cv2.rectangle(
                display_frame,
                (wx1, wy1),
                (wx2, wy2),
                (0, 0, 255),
                2
            )

            cv2.circle(
                display_frame,
                (cx, cy),
                5,
                (0, 255, 255),
                -1
            )

            label = (
                f"Wound {wound_conf:.0%} | "
                f"P{person['id']} | "
                f"{body_part}"
            )

            cv2.putText(
                display_frame,
                label,
                (
                    wx1,
                    max(
                        20,
                        wy1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                2
            )

            # Show the original class from best.pt
            cv2.putText(
                display_frame,
                f"Model class: {model_class}",
                (
                    wx1,
                    min(
                        FRAME_HEIGHT - 10,
                        wy2 + 18
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1
            )

    return accepted_wounds
