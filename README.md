# EagleVision Pipeline - Technical Assessment

## Overview
EagleVision is an AI-powered pipeline designed to monitor and analyze construction equipment using computer vision. It processes video feeds to track equipment (e.g., excavators, trucks), classify their activities, and rigorously monitor idle (dwell) time to measure productivity. The analyzed data is streamed seamlessly via Apache Kafka to be consumed by downstream analytics and visualization dashboards.

This project was built focusing on **System Architecture, Real-time Reliability, and Extensibility**. 

## Features
- **Object Detection & Tracking:** Uses YOLOv8s and ByteTrack for robust object tracking.
- **Occlusion-Resistant Re-ID:** Implements both Visual Feature Extraction (Cosine Similarity) and Spatial Fallback to maintain equipment identity even if lost behind obstacles or dirt.
- **Motion Analytics:** Calculates dense optical flow (Farneback) on sub-regions to determine articulated motion state (e.g. arm moving while the chassis is static).
- **Activity Classification:** Differentiates between SWINGING, DIGGING, DUMPING, WAITING, and MOVING.
- **Robust Dwell Time Tracker:** A state machine ensuring non-stop accumulation of `Total Idle Time` across tracking losses and maintaining `Current Session` timers gracefully.
- **Microservices Data Streaming:** Streams JSON payloads of the analyzed state locally via `stdout` and remotely via Kafka topics to decouple computer vision loads from dashboard/UI analytics.

## Project Structure
```text
├── cv_service/
│   ├── main.py                   # Main Video CV loop
│   ├── detector.py               # Ultralytics bounding box wrapping
│   ├── motion_analyzer.py        # Sub-regional Optical Flow Logic
│   ├── activity_classifier.py    # Rule-based action interpretation
│   ├── dwell_time_tracker.py     # State management & Idle Time accounting
│   ├── reid_module.py            # Visual/Spatial Matching for Lost Tracks
│   └── kafka_producer.py         # Producer implementation
├── ui_service/
│   └── app.py                    # Streamlit Dashboard consuming Kafka
├── docker-compose.yml            # Zookeeper/Kafka/Database startup
└── requirements.txt              # Dependencies (YOLO, OpenCV, Kafka, Streamlit)
```

## System Requirements
- Python 3.10+
- `pip install ultralytics opencv-python numpy kafka-python streamlit`
- Docker & Docker-Compose (For Kafka Infrastructure)

## Quick Start (Local Evaluation)

### 1. View Computer Vision Module (Standalone)
You can run the CV module entirely standalone to evaluate tracking, performance, and print the Kafka-ready JSON directly to your terminal.
```bash
cd cv_service
python main.py --source ../data/test_video.mp4 --show
```
*Note: Depending on your hardware, processing may run between 5-30 FPS. The terminal will log exact rendering FPS and Estimated Time remaining.*

### 2. Full Microservices Pipeline (Kafka + Streamlit UI)
To see the full architectural separation in action:

**Terminal 1 (Infrastructure):**
```bash
docker-compose up -d
```

**Terminal 2 (Start Data Dashboard):**
```bash
cd ui_service
streamlit run app.py
```

**Terminal 3 (Start CV Engine streaming to Kafka):**
```bash
cd cv_service
python main.py --source ../data/test_video.mp4 --kafka localhost:9092
```

## Technical Write-up: Design Decisions & Challenges

### 1. The Articulated Equipment Challenge (Arm-only Motion)
One of the key requirements was detecting when an excavator is "working" even if its base (chassis) is stationary. 
**Our Solution:** Instead of a simple global motion check, we implemented a **Spatial Sub-Region Analysis** in `motion_analyzer.py`.
- We split the detected bounding box into a top region (Arm/Bucket) and a bottom region (Tracks/Chassis).
- We calculate the **Dense Optical Flow (Farneback)** separately for each.
- **Logic:** If `top_flow > threshold` and `bottom_flow < threshold`, we classify it as **Articulated Motion**. This allows the system to distinguish between a vehicle that's just "Waiting" and one that's "Digging" while stationary.

### 2. Activity Classification Logic
We used a rule-based engine in `activity_classifier.py` that processes the motion vectors ($dx, dy$):
- **Digging:** High downward vertical flow ($dy > 0.2$).
- **Dumping:** High upward vertical flow ($dy < -0.2$).
- **Swinging:** High horizontal flow ($|dx| / |dy| > 1.3$).
- **Waiting:** Overall flow magnitude $< 0.15$.
- **Temporal Smoothing:** We use a `deque` based moving average (3-frame window) to debounce noise and ensure stable state transitions.

### 3. Re-ID & Occlusion Handling
To solve the "identity reset" problem during occlusions, we built a hybrid Re-ID module:
- **Visual Embedding:** A 256-dim feature vector (Color + Texture) is extracted for every track.
- **Spatial Fallback:** If a vehicle is lost and a "new" one appears in the same pixel vicinity (dead-reckoning), we restore the ID and its non-stop `Total Idle Time` counter.

## Known Limitations & Production Roadmap
- **Detection during heavy occlusions:** This uses `yolov8s` pre-trained on COCO dataset. **Production Fix:** Fine-tune on a specialized Construction Dataset.
- **Microservices Scaling:** Currently uses a single Kafka broker. **Production Fix:** Cluster-based Kafka with a dedicated database (PostgreSQL/TimescaleDB) for historical analytics.
