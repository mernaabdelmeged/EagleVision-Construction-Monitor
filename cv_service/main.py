"""
main.py — Entry point for CV Service
====================================
Integrates detection, tracking, Re-ID, motion analysis, activity classification,
and Dwell Time tracking into a single processing loop.
"""

import cv2
import json
import os
import time
import argparse
from ultralytics import YOLO

from reid_module import ReIDModule
from motion_analyzer import MotionAnalyzer
from activity_classifier import ActivityClassifier
from dwell_time_tracker import DwellTimeTracker
from kafka_producer import EquipmentEventProducer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="../data/test_video.mp4", help="Path to video")
    parser.add_argument("--kafka", type=str, default="None", help="Kafka bootstrap server")
    parser.add_argument("--show", action="store_true", help="Show video visualization")
    return parser.parse_args()

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def main():
    args = parse_args()

    # -- 0. Initial Cleanup --
    # Ensure UI resets to zero by clearing the old snapshot file
    snapshot_path = "../data/live_snapshot.json"
    if os.path.exists(snapshot_path):
        try:
            os.remove(snapshot_path)
        except:
            pass

    # 1. Initialize YOLOv8 with ByteTrack built-in
    model = YOLO("yolov8s.pt")
    
    # 2. Initialize our modules
    reid_module = ReIDModule(max_lost_frames=18000, similarity_threshold=0.4)
    # Calibrated for "second-by-second" ultra-responsiveness
    motion_analyzer = MotionAnalyzer(arm_threshold=0.3, body_threshold=0.2)
    activity_classifier = ActivityClassifier(smoothing_window=3)
    dwell_tracker = DwellTimeTracker()
    
    # Optional Kafka
    kafka = EquipmentEventProducer(bootstrap_servers=args.kafka) if args.kafka != "None" else None
    
    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error opening video: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    print(f"Starting Video Processing... Total Frames to process: {total_frames}")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        timestamp = frame_idx / fps
        
        # Added class 2 (car) because YOLO sometimes misclassifies obscured construction equipment as cars.
        results = model.track(frame, persist=True, classes=[2, 5, 7], conf=0.1, verbose=False)
        
        vis_frame = frame.copy()
        
        current_tracked_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            valid_boxes = []
            
            for bbox, t_id, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
                x1, y1, x2, y2 = map(int, bbox)
                
                if x2 - x1 < 40 or y2 - y1 < 40:
                    continue
                    
                # Suppress ghost duplicate boxes from ByteTrack! 
                # If a box massively overlaps (IoU > 60%) with one we already tracked, kill it.
                is_duplicate = False
                for v_box in valid_boxes:
                    if bb_iou((x1, y1, x2, y2), v_box) > 0.6:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                valid_boxes.append((x1, y1, x2, y2))
                
                class_name = model.names[cls_id]
                if class_name == "bus":
                    class_name = "heavy_equipment"
                else:
                    class_name = "truck"
                    
                # -- 1. Re-ID Module --
                # Check if this track is already an active continuous track
                resolved_id = reid_module.resolve_id(t_id)
                
                if resolved_id in reid_module.active_gallery:
                    # It's an existing active track, no need to search the lost gallery
                    is_reidentified = False
                else:
                    # New track ID! Let's see if we can match it to a lost one
                    resolved_id, is_reidentified, _ = reid_module.try_reidentify(
                        frame, t_id, [x1, y1, x2, y2], class_name
                    )
                
                reid_module.update_active(frame, resolved_id, [x1, y1, x2, y2])
                current_tracked_ids.add(resolved_id)
                
                eq_id_str = f"EQ-{resolved_id}"
                
                # -- 2. Motion Analysis --
                motion = motion_analyzer.analyze(frame, resolved_id, [x1, y1, x2, y2])
                
                # -- 3. Activity Classification --
                activity_res = activity_classifier.classify(
                    resolved_id, 
                    motion.state.value, 
                    motion.motion_source,
                    motion.flow_dx,
                    motion.flow_dy,
                    motion.overall_flow_mag
                )
                
                # -- 4. Dwell Time Tracking --
                # Use the SMOOTHED activity state to determine ACTIVE/INACTIVE 
                # instead of raw motion state which flickers every single frame!
                smoothed_state = "INACTIVE" if activity_res.activity.value == "WAITING" else "ACTIVE"
                
                machine_state = dwell_tracker.update(
                    equipment_id=eq_id_str,
                    equipment_class=class_name,
                    current_state=smoothed_state,
                    current_activity=activity_res.activity.value,
                    motion_source=motion.motion_source,
                    timestamp=timestamp,
                    is_reidentified=is_reidentified
                )
                
                # -- 5. Visualization --
                color = (0, 255, 0) if machine_state.current_state == "ACTIVE" else (0, 0, 255)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{eq_id_str} | {activity_res.activity.value}"
                # Background rectangle for the top label to make it readable even if overlapping
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 0), -1)
                cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                idle_now = DwellTimeTracker.format_seconds(machine_state.current_session_idle_sec)
                idle_total = DwellTimeTracker.format_seconds(machine_state.total_idle_sec)
                
                # Draw Now and Total inside the bottom of the bounding box!
                # This separates them from the top label and reduces collision when trucks overlap.
                # Black outline (stroke)
                cv2.putText(vis_frame, f"Total: {idle_total}", (x1 + 5, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4)
                cv2.putText(vis_frame, f"Now: {idle_now}", (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4)
                
                # White text inside
                cv2.putText(vis_frame, f"Total: {idle_total}", (x1 + 5, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(vis_frame, f"Now: {idle_now}", (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                
        # -- 6. Live Data Streaming & Snapshots (Every 30 frames ~ 1 sec) --
        if frame_idx % 30 == 0:
            system_snapshot = {}
            for m_id, m_state in dwell_tracker.get_all_machines().items():
                # Format timestamp as HH:MM:SS.mmm
                ts_ms = int((frame_idx / 30.0) * 1000)
                h = ts_ms // 3600000
                m = (ts_ms % 3600000) // 60000
                s = (ts_ms % 60000) // 1000
                ms = ts_ms % 1000
                ts_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
                
                system_snapshot[m_id] = {
                    "frame_id": frame_idx,
                    "equipment_id": m_id,
                    "equipment_class": m_state.equipment_class,
                    "timestamp": ts_str,
                    "utilization": {
                        "current_state": m_state.current_state,
                        "current_activity": m_state.current_activity,
                        "motion_source": m_state.motion_source
                    },
                    "time_analytics": {
                        "total_tracked_seconds": round(m_state.total_tracked_sec, 1),
                        "total_active_seconds": round(m_state.total_active_sec, 1),
                        "total_idle_seconds": round(m_state.total_idle_sec, 1),
                        "utilization_percent": m_state.utilization_percent
                    }
                }
            
            # Write full snapshot for Zero-Latency Dashboard
            os.makedirs("../data", exist_ok=True)
            with open("../data/live_snapshot.json", "w") as f:
                json.dump(system_snapshot, f)
            
            # Emit to Kafka (if enabled)
            if kafka:
                for event in system_snapshot.values():
                    kafka.send_event(event)
            
            # Log progress with professional ETA
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0
            frames_left = total_frames - frame_idx
            eta_min = (frames_left / current_fps) / 60 if current_fps > 0 else 0
            print(f"\n🚀 [SNAPSHOT] Frame {frame_idx}/{total_frames} | Speed: {current_fps:.1f} FPS | ETA: {eta_min:.1f} min")
            
            # Print the JSON Kafka Payload (visible in demo video)
            for eq_event in system_snapshot.values():
                print(json.dumps(eq_event, indent=2))
            print("─" * 60)
        # -- Re-ID Maintenance --
        # Find tracks that disappeared in this frame
        active_track_ids = set(reid_module.active_gallery.keys())
        lost_ids = active_track_ids - current_tracked_ids
        for lost_id in lost_ids:
            # Save dwell state before moving to lost gallery
            eq_id_str = f"EQ-{lost_id}"
            state_data = dwell_tracker.export_state(eq_id_str)
            reid_module.mark_lost(lost_id, dwell_state=state_data)
            
            # NOTE: We intentionally DO NOT call remove_track() on activity_classifier 
            # and motion_analyzer! Keeping their historical memory preserves the 
            # smoothing queue, so when Re-ID restores the ID, the vehicle doesn't 
            # violently jump to 'ACTIVE' state and reset the 'Now' timer.
            
        reid_module.tick()

        # Render display
        if args.show:
            cv2.imshow("EagleVision CV Pipeline", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    if kafka:
        kafka.close()
    
    print("\nProcessing Complete!")
    print("\n--- Final Dwell Time Report ---")
    for eq_id, state in dwell_tracker.get_all_machines().items():
        print(f"{eq_id} ({state.equipment_class}):")
        print(f"  Total Tracked: {state.total_tracked_sec:.1f}s")
        print(f"  Total Idle:    {state.total_idle_sec:.1f}s")
        print(f"  Utilization:   {state.utilization_percent}%")
        print(f"  Re-identified: {state.reidentified_count} times")

if __name__ == "__main__":
    main()
