# NAJEED Drone AI

NAJEED is an AI-based emergency response drone system developed using the RoboMaster TT / DJI Tello drone. The system analyzes the live drone camera stream to detect people, assess their movement and consciousness status, detect visible wounds, and identify the affected body area.

The project combines pose estimation, wound segmentation, person tracking, real-time video analysis, and drone control in one integrated system.

---

## Project Features

- Real-time drone camera streaming
- Person detection and tracking
- Human pose estimation
- Conscious and unconscious status detection
- Body movement analysis
- Wound detection and segmentation
- Identification of the injured body part
- Multiple person tracking
- Bounding box stabilization
- Real-time drone control
- Video recording of the processed stream

---

## Project Structure

```text
NAJEED_Drone/
│
├── pose_consciousness.py
├── wound_detection.py
├── drone_health.py
├── yolo11n-pose.pt
└── best.pt
```

### `pose_consciousness.py`

The main project file.

It handles:

- RoboMaster TT connection
- Camera streaming
- Person detection
- Pose estimation
- Person tracking
- Body movement analysis
- Conscious and unconscious classification
- Integration with wound detection
- Drone movement controls
- Video recording

Run this file to start the complete NAJEED system.

### `wound_detection.py`

Contains the wound detection and processing functions.

It handles:

- Wound segmentation
- Wound bounding boxes
- Wound masks
- Assigning detected wounds to the correct person
- Identifying the affected body part
- Filtering invalid wound detections

This file is called automatically by `pose_consciousness.py`.

### `drone_health.py`

Used to check the drone status before or during operation.

It monitors information such as:

- Battery level
- Drone temperature
- Connection status

### `yolo11n-pose.pt`

A pretrained YOLO11 Nano Pose model used for:

- Person detection
- Human pose estimation
- Extraction of 17 body keypoints
- Body movement analysis

### `best.pt`

The trained wound segmentation model used by NAJEED.

The model was fine-tuned from:

```text
yolo26n-seg.pt
```

It is used during inference to detect and segment visible wounds.

---

## Consciousness Detection

The system monitors the person's body movement using pose keypoints and waits for approximately **5 seconds before determining the consciousness status**.

During these 5 seconds, the system continuously checks for confirmed body movement.

If movement is detected, the person is classified as:

```text
CONSCIOUS
```

If no confirmed movement is detected for **5 continuous seconds**, the system classifies the person as:

```text
UNCONSCIOUS
```

The movement calculation mainly uses body keypoints while excluding facial keypoints because facial keypoints may produce small unstable movements even when the person is actually still.

The system also applies movement smoothing and requires movement to be confirmed across multiple frames before considering it actual body movement.

---

## Wound Detection

The trained segmentation model detects visible wounds from the drone camera stream.

For each accepted wound detection, the system:

1. Detects the wound using `best.pt`
2. Extracts the wound segmentation mask
3. Associates the wound with the correct detected person
4. Calculates the wound location
5. Compares the wound position with the person's pose keypoints
6. Determines the affected body area

Possible body areas include:

```text
HEAD
TORSO
LEFT ARM
RIGHT ARM
LEFT LEG
RIGHT LEG
```

---

## Models

| Model | Purpose |
|---|---|
| `yolo11n-pose.pt` | Human pose estimation and body keypoint detection |
| `best.pt` | Fine-tuned wound segmentation model |

---

## Requirements

Install the required Python libraries:

```bash
pip install ultralytics
pip install djitellopy
pip install opencv-python
pip install numpy
```

---

## Hardware

The project is designed for:

```text
RoboMaster TT / DJI Tello
```

The drone provides the live camera stream while the AI models run on the connected computer.

No additional AI hardware is required on the drone.

---

## Running the Project

Place all project files and model files in the same directory:

```text
pose_consciousness.py
wound_detection.py
drone_health.py
yolo11n-pose.pt
best.pt
```

Connect the computer to the RoboMaster TT / Tello Wi-Fi network.

Then run:

```bash
python pose_consciousness.py
```

The main script loads the pose model and calls the wound detection functions automatically.

---

## Drone Controls

| Key | Action |
|---|---|
| `Space` | Takeoff |
| `L` | Land |
| `R` | Rotate 360 degrees |
| `W` | Move forward |
| `S` | Move backward |
| `A` | Move left |
| `D` | Move right |
| `I` | Move up |
| `K` | Move down |
| `E` | Rotate right |
| `Q` | Rotate left |
| `ESC` | Exit and stop the system |

---

## Video Output

The processed drone stream is recorded automatically.

The output video is saved as:

```text
drone_medical_rescue.mp4
```

The recorded video can include:

- Detected people
- Pose skeletons
- Person IDs
- Movement status
- Consciousness status
- Wound detections
- Wound locations
- Affected body parts

---

## System Workflow

```text
Drone Camera
     |
     v
Person Detection
     |
     v
Pose Estimation
     |
     +-----------------------------+
     |                             |
     v                             v
Movement Analysis             Wound Detection
     |                             |
     v                             v
5-Second Monitoring          Wound Segmentation
     |                             |
     v                             v
Consciousness Status       Assign Wound to Person
                                   |
                                   v
                           Body Part Localization
                                   |
                                   v
                              Final Output
```

---

## Technologies

- Python
- Ultralytics YOLO
- YOLO11 Pose
- YOLO Segmentation
- OpenCV
- NumPy
- DJITelloPy
- RoboMaster TT / DJI Tello

---

## Project Overview

NAJEED was developed as an AI-assisted drone system for emergency response and preliminary casualty assessment.

The system uses real-time computer vision to identify people, monitor their movement, determine whether they are conscious or unconscious after a five-second observation period, detect visible wounds, and identify the affected body area through the drone camera stream.
