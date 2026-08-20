import cv2
import time
import numpy as np
from pathlib import Path
from djitellopy import Tello
from ultralytics import YOLO

from wound_detection import load_wound_model, process_wounds

# =========================================================
# SETTINGS
# =========================================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Pose
POSE_CONFIDENCE = 0.55
POSE_IOU = 0.55
KEYPOINT_CONFIDENCE = 0.35

# Consciousness
STILL_TIME_LIMIT = 5.0
MOTION_THRESHOLD = 6.0
MOTION_CONFIRM_FRAMES = 2

LOW_MOVEMENT = 3.0
MEDIUM_MOVEMENT = 8.0
HIGH_MOVEMENT = 15.0

# Stable person tracking / duplicate removal
DUPLICATE_IOU_THRESHOLD = 0.65
TRACK_MATCH_IOU = 0.15
TRACK_MATCH_CENTER_DISTANCE = 120
TRACK_MAX_MISSING_FRAMES = 20
SMOOTH_FACTOR = 0.70

# Flight
FLIGHT_SPEED = 40

BASE_DIR = Path(__file__).resolve().parent

SKELETON_EDGES = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# =========================================================
# LOCAL MODEL LOADING
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

def smooth_box(old_box, new_box):
    new_box = np.asarray(new_box, dtype=float)

    if old_box is None:
        return new_box.copy()

    old_box = np.asarray(old_box, dtype=float)

    return (
        old_box * SMOOTH_FACTOR
        + new_box * (1.0 - SMOOTH_FACTOR)
    )

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def box_center(box):
    x1, y1, x2, y2 = map(float, box)

    return np.array(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        dtype=float
    )

def remove_duplicate_people(boxes, scores):
    """
    Keep one person detection when YOLO gives overlapping detections.
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    for idx in order:
        duplicate = False

        for kept_idx in keep:
            if (
                box_iou(boxes[idx], boxes[kept_idx])
                >= DUPLICATE_IOU_THRESHOLD
            ):
                duplicate = True
                break

        if not duplicate:
            keep.append(int(idx))

    return keep

def match_to_existing_track(det_box, tracks, used_ids):
    """
    Simple persistent ID matching using box IoU + center distance.
    """
    best_id = None
    best_score = -1e9

    det_center = box_center(det_box)

    for track_id, state in tracks.items():

        if track_id in used_ids:
            continue

        old_box = state["raw_box"]

        iou = box_iou(det_box, old_box)

        center_distance = np.linalg.norm(
            det_center - box_center(old_box)
        )

        if (
            iou < TRACK_MATCH_IOU
            and center_distance > TRACK_MATCH_CENTER_DISTANCE
        ):
            continue

        score = iou * 1000.0 - center_distance

        if score > best_score:
            best_score = score
            best_id = track_id

    return best_id

def draw_pose(frame, xy, conf):

    for start_idx, end_idx in SKELETON_EDGES:

        if (
            conf[start_idx] >= KEYPOINT_CONFIDENCE
            and conf[end_idx] >= KEYPOINT_CONFIDENCE
        ):
            p1 = tuple(xy[start_idx].astype(int))
            p2 = tuple(xy[end_idx].astype(int))

            cv2.line(
                frame,
                p1,
                p2,
                (255, 255, 255),
                2
            )

    for idx in range(len(xy)):

        if conf[idx] >= KEYPOINT_CONFIDENCE:

            x, y = xy[idx].astype(int)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 255),
                -1
            )

def calculate_movement(
    current_xy,
    current_conf,
    previous_xy,
    previous_conf
):
    """
    Calculate body movement using body keypoints only.
    Face points are ignored because they jitter easily.
    """

    body_indices = np.arange(5, 17)

    valid = (
        (current_conf[body_indices] >= KEYPOINT_CONFIDENCE)
        &
        (previous_conf[body_indices] >= KEYPOINT_CONFIDENCE)
    )

    valid_indices = body_indices[valid]

    if len(valid_indices) < 4:
        return None

    distances = np.linalg.norm(
        current_xy[valid_indices]
        - previous_xy[valid_indices],
        axis=1
    )

    # Median reduces the effect of one badly jumping keypoint
    return float(np.median(distances))

def movement_label(movement):

    if movement is None:
        return "NO MOVEMENT"

    if movement >= HIGH_MOVEMENT:
        return "HIGH"

    if movement >= MEDIUM_MOVEMENT:
        return "MEDIUM"

    if movement >= LOW_MOVEMENT:
        return "LOW"

    return "NO MOVEMENT"


# =========================================================
# MAIN
# =========================================================
def main():

    # Try local files only.
    pose_model_path = find_local_model(
        [
            "yolo11n-pose.pt"
        ]
    )

    print(
        f"Loading Pose model: "
        f"{Path(pose_model_path).name}"
    )
    pose_model = YOLO(
        pose_model_path
    )

    wound_model = load_wound_model()

    tello = Tello()

    stream_started = False
    is_flying = False
    video_writer = None

    tracks = {}
    next_track_id = 1

    try:

        print(
            "Connecting to RoboMaster TT..."
        )

        tello.connect()

        print(
            f"Battery: "
            f"{tello.get_battery()}%"
        )

        tello.streamon()
        stream_started = True

        frame_read = (
            tello.get_frame_read()
        )

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        video_writer = (
            cv2.VideoWriter(
                "drone_medical_rescue.mp4",
                fourcc,
                20.0,
                (
                    FRAME_WIDTH,
                    FRAME_HEIGHT
                )
            )
        )

        print(
            "==================================="
        )
        print("SPACE = Takeoff")
        print("L = Land")
        print("R = Rotate 360")
        print("W/S = Forward / Back")
        print("A/D = Left / Right")
        print("I/K = Up / Down")
        print("E/Q = Rotate Right / Left")
        print("ESC = Exit")
        print(
            "==================================="
        )

        while True:

            frame = frame_read.frame

            if frame is None:
                continue

            # Fix RoboMaster TT / Tello camera colors for OpenCV
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            frame = cv2.resize(
                frame,
                (
                    FRAME_WIDTH,
                    FRAME_HEIGHT
                )
            )

            display_frame = (
                frame.copy()
            )

            now = time.time()

            # =================================================
            # 1. POSE MODEL
            # =================================================
            pose_results = (
                pose_model.predict(
                    source=frame,
                    conf=POSE_CONFIDENCE,
                    iou=POSE_IOU,
                    verbose=False
                )
            )

            pose_result = (
                pose_results[0]
            )

            boxes = np.empty(
                (0, 4),
                dtype=float
            )

            scores = np.empty(
                (0,),
                dtype=float
            )

            kp_xy = np.empty(
                (0, 17, 2),
                dtype=float
            )

            kp_conf = np.empty(
                (0, 17),
                dtype=float
            )

            if (
                pose_result.boxes
                is not None
                and len(
                    pose_result.boxes
                ) > 0
            ):
                boxes = (
                    pose_result
                    .boxes
                    .xyxy
                    .cpu()
                    .numpy()
                )

                scores = (
                    pose_result
                    .boxes
                    .conf
                    .cpu()
                    .numpy()
                )

            if (
                pose_result.keypoints
                is not None
                and
                pose_result.keypoints.data
                is not None
            ):
                kp_data = (
                    pose_result
                    .keypoints
                    .data
                    .cpu()
                    .numpy()
                )

                if len(kp_data) > 0:

                    kp_xy = (
                        kp_data[:, :, :2]
                    )

                    kp_conf = (
                        kp_data[:, :, 2]
                    )

            count = min(
                len(boxes),
                len(kp_xy)
            )

            if count > 0:

                boxes = boxes[:count]
                scores = scores[:count]
                kp_xy = kp_xy[:count]
                kp_conf = kp_conf[:count]

                keep_indices = (
                    remove_duplicate_people(
                        boxes,
                        scores
                    )
                )

            else:
                keep_indices = []

            # =================================================
            # 2. STABLE PERSON IDs + CONSCIOUSNESS
            # =================================================
            people = []
            used_track_ids = set()

            for det_idx in keep_indices:

                det_box = (
                    boxes[det_idx]
                )

                current_xy = (
                    kp_xy[det_idx]
                )

                current_conf = (
                    kp_conf[det_idx]
                )

                track_id = (
                    match_to_existing_track(
                        det_box,
                        tracks,
                        used_track_ids
                    )
                )

                if track_id is None:

                    track_id = (
                        next_track_id
                    )

                    next_track_id += 1

                    tracks[track_id] = {
                        "raw_box":
                            det_box.copy(),

                        "smooth_box":
                            det_box.copy(),

                        "previous_xy":
                            None,

                        "previous_conf":
                            None,

                        "status":
                            "CONSCIOUS",

                        "last_confirmed_motion":
                            now,

                        "motion_streak":
                            0,

                        "movement_ema":
                            0.0,

                        "missing_frames":
                            0,
                    }

                state = (
                    tracks[track_id]
                )

                used_track_ids.add(
                    track_id
                )

                state[
                    "missing_frames"
                ] = 0

                state[
                    "raw_box"
                ] = det_box.copy()

                state[
                    "smooth_box"
                ] = smooth_box(
                    state["smooth_box"],
                    det_box
                )

                movement = None

                if (
                    state[
                        "previous_xy"
                    ] is not None
                    and
                    state[
                        "previous_conf"
                    ] is not None
                ):
                    movement = (
                        calculate_movement(
                            current_xy,
                            current_conf,
                            state[
                                "previous_xy"
                            ],
                            state[
                                "previous_conf"
                            ]
                        )
                    )

                if movement is not None:

                    state[
                        "movement_ema"
                    ] = (
                        0.65
                        * state[
                            "movement_ema"
                        ]
                        +
                        0.35
                        * movement
                    )

                    if (
                        state[
                            "movement_ema"
                        ]
                        >= MOTION_THRESHOLD
                    ):
                        state[
                            "motion_streak"
                        ] += 1

                    else:
                        state[
                            "motion_streak"
                        ] = 0

                    if (
                        state[
                            "motion_streak"
                        ]
                        >= MOTION_CONFIRM_FRAMES
                    ):
                        state[
                            "last_confirmed_motion"
                        ] = now

                        state[
                            "status"
                        ] = "CONSCIOUS"

                still_seconds = (
                    now
                    - state[
                        "last_confirmed_motion"
                    ]
                )

                if (
                    still_seconds
                    >= STILL_TIME_LIMIT
                ):
                    state[
                        "status"
                    ] = "UNCONSCIOUS"

                state[
                    "previous_xy"
                ] = current_xy.copy()

                state[
                    "previous_conf"
                ] = current_conf.copy()

                people.append(
                    {
                        "id":
                            track_id,

                        "box":
                            state[
                                "smooth_box"
                            ].copy(),

                        "xy":
                            current_xy.copy(),

                        "conf":
                            current_conf.copy(),

                        "status":
                            state[
                                "status"
                            ],

                        "movement":
                            movement_label(
                                state[
                                    "movement_ema"
                                ]
                            ),
                    }
                )

            # =================================================
            # 3. CLEAN LOST TRACKS
            # =================================================
            for track_id in list(
                tracks.keys()
            ):

                if (
                    track_id
                    not in used_track_ids
                ):

                    tracks[
                        track_id
                    ][
                        "missing_frames"
                    ] += 1

                    if (
                        tracks[
                            track_id
                        ][
                            "missing_frames"
                        ]
                        >
                        TRACK_MAX_MISSING_FRAMES
                    ):
                        tracks.pop(
                            track_id,
                            None
                        )

            # =================================================
            # 4. DRAW PERSONS
            # =================================================
            for person in people:

                track_id = (
                    person["id"]
                )

                x1, y1, x2, y2 = (
                    person["box"]
                    .astype(int)
                )

                current_xy = (
                    person["xy"]
                )

                current_conf = (
                    person["conf"]
                )

                status = (
                    person["status"]
                )

                movement = (
                    person["movement"]
                )

                draw_pose(
                    display_frame,
                    current_xy,
                    current_conf
                )

                if (
                    status
                    == "CONSCIOUS"
                ):
                    status_color = (
                        0,
                        255,
                        0
                    )
                else:
                    status_color = (
                        0,
                        0,
                        255
                    )

                # Only one person box
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    status_color,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"P{track_id}: {status}",
                    (
                        x1,
                        max(
                            y1 - 42,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    status_color,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Movement: {movement}",
                    (
                        x1,
                        max(
                            y1 - 20,
                            42
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )

            # =================================================
            # 5. ACTUAL TRAINED WOUND SEGMENTATION MODEL
            # =================================================
            accepted_wounds = process_wounds(
                frame,
                display_frame,
                people,
                wound_model
            )

            # =================================================
            # 6. COUNTERS
            # =================================================
            cv2.putText(
                display_frame,
                f"People: {len(people)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Wounds: {accepted_wounds}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 255),
                2
            )

            # =================================================
            # 7. SAVE + SHOW
            # =================================================
            if video_writer is not None:

                video_writer.write(
                    display_frame
                )

            cv2.imshow(
                "RoboMaster TT - Pose + Wound AI",
                display_frame
            )

            # =================================================
            # 8. KEYBOARD CONTROL
            # =================================================
            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            lr = 0
            fb = 0
            ud = 0
            yaw = 0

            if key == 32:

                if not is_flying:

                    tello.takeoff()
                    is_flying = True

            elif key in (
                ord("l"),
                ord("L")
            ):

                if is_flying:

                    tello.land()
                    is_flying = False

            elif key in (
                ord("r"),
                ord("R")
            ):

                if is_flying:

                    tello.rotate_clockwise(
                        360
                    )

            elif key in (
                ord("w"),
                ord("W")
            ):
                fb = FLIGHT_SPEED

            elif key in (
                ord("s"),
                ord("S")
            ):
                fb = -FLIGHT_SPEED

            elif key in (
                ord("a"),
                ord("A")
            ):
                lr = -FLIGHT_SPEED

            elif key in (
                ord("d"),
                ord("D")
            ):
                lr = FLIGHT_SPEED

            elif key in (
                ord("i"),
                ord("I")
            ):
                ud = FLIGHT_SPEED

            elif key in (
                ord("k"),
                ord("K")
            ):
                ud = -FLIGHT_SPEED

            elif key in (
                ord("e"),
                ord("E")
            ):
                yaw = FLIGHT_SPEED

            elif key in (
                ord("q"),
                ord("Q")
            ):
                yaw = -FLIGHT_SPEED

            elif key == 27:

                if is_flying:

                    tello.land()
                    is_flying = False

                break

            if is_flying:

                tello.send_rc_control(
                    lr,
                    fb,
                    ud,
                    yaw
                )

    finally:

        if is_flying:

            try:
                tello.land()
            except Exception:
                pass

        if stream_started:

            try:
                tello.streamoff()
            except Exception:
                pass

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()

        try:
            tello.end()
        except Exception:
            pass

        print(
            "Saved: drone_medical_rescue.mp4"
        )


if __name__ == "__main__":
    main()
