"""
activity_classifier.py — Rule-based activity classification
===========================================================
Classifies equipment activity from optical flow direction + state.

Activities:
    DIGGING    — arm moving DOWN into ground
    SWINGING   — arm moving laterally (LEFT or RIGHT)
    DUMPING    — arm moving UP and outward
    WAITING    — equipment INACTIVE (not moving)
    MOVING     — whole body translating (driving)
"""

import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass


class Activity(str, Enum):
    DIGGING = "DIGGING"
    SWINGING = "SWINGING"
    DUMPING = "DUMPING"
    WAITING = "WAITING"
    MOVING = "MOVING"
    UNKNOWN = "UNKNOWN"


@dataclass
class ActivityResult:
    activity: Activity
    confidence: float
    flow_dx: float
    flow_dy: float


class ActivityClassifier:
    """
    Rule-based activity classification using optical flow direction.
    
    Uses a short temporal window to smooth predictions and reduce noise.
    
    Flow direction interpretation:
        dy < 0  (upward flow)    → DUMPING  (arm raising)
        dy > 0  (downward flow)  → DIGGING  (arm lowering into ground)
        |dx| >> |dy|             → SWINGING (arm rotating left/right)
        full body moving         → MOVING   (vehicle driving)
        no flow                  → WAITING
    """

    def __init__(self, smoothing_window: int = 10):
        # Per-track activity history for temporal smoothing
        self._history: dict[int, deque] = {}
        self.smoothing_window = smoothing_window

        # Thresholds tuned for ultra-responsive "second-by-second" detection
        self.MIN_FLOW_FOR_ACTIVITY = 0.15  # Catch ultra-slow digging/loading
        self.SWING_DOMINANCE_RATIO = 1.3   # |dx| / |dy| > this → SWINGING
        self.DUMP_DY_THRESHOLD = -0.2      # dy < this → DUMPING (upward)
        self.DIG_DY_THRESHOLD = 0.2        # dy > this → DIGGING (downward)

    def classify(
        self,
        track_id: int,
        motion_state: str,          # "ACTIVE" or "INACTIVE"
        motion_source: str,         # "arm_only", "full_body", "body_only", "none"
        flow_dx: float,
        flow_dy: float,
        overall_flow_mag: float,
    ) -> ActivityResult:
        """
        Classify current activity for a tracked equipment.
        
        Returns smoothed ActivityResult.
        """
        # Base classification
        raw_activity = self._classify_raw(
            motion_state, motion_source, flow_dx, flow_dy, overall_flow_mag
        )

        # Temporal smoothing
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.smoothing_window)
        self._history[track_id].append(raw_activity)

        # Majority vote from history
        smoothed = self._majority_vote(self._history[track_id])

        return ActivityResult(
            activity=smoothed,
            confidence=self._compute_confidence(self._history[track_id], smoothed),
            flow_dx=flow_dx,
            flow_dy=flow_dy,
        )

    def _classify_raw(
        self,
        motion_state: str,
        motion_source: str,
        flow_dx: float,
        flow_dy: float,
        overall_flow_mag: float,
    ) -> Activity:
        """Single-frame classification (before smoothing)."""

        # No motion → WAITING
        if motion_state == "INACTIVE" or overall_flow_mag < self.MIN_FLOW_FOR_ACTIVITY:
            return Activity.WAITING

        # Full body moving (driving/repositioning) → MOVING
        if motion_source == "body_only":
            return Activity.MOVING

        # Arm-only or full body — classify by direction
        abs_dx = abs(flow_dx)
        abs_dy = abs(flow_dy)

        # Lateral dominance → SWINGING
        if abs_dx > 0.1 and abs_dx / (abs_dy + 0.001) > self.SWING_DOMINANCE_RATIO:
            return Activity.SWINGING

        # Upward flow → DUMPING (arm raising to dump material)
        if flow_dy < self.DUMP_DY_THRESHOLD:
            return Activity.DUMPING

        # Downward flow → DIGGING (arm pushing into ground)
        if flow_dy > self.DIG_DY_THRESHOLD:
            return Activity.DIGGING

        # Mixed motion → default to SWINGING if arm is moving
        if motion_source in ("arm_only", "full_body"):
            return Activity.SWINGING

        return Activity.UNKNOWN

    def _majority_vote(self, history: deque) -> Activity:
        """Return most frequent activity in history."""
        if not history:
            return Activity.WAITING
        counts = {}
        for act in history:
            counts[act] = counts.get(act, 0) + 1
        return max(counts, key=counts.get)

    def _compute_confidence(self, history: deque, dominant: Activity) -> float:
        """Compute confidence as fraction of frames with dominant activity."""
        if not history:
            return 0.0
        count = sum(1 for a in history if a == dominant)
        return count / len(history)

    def remove_track(self, track_id: int):
        """Clean up when track is removed."""
        self._history.pop(track_id, None)

    @staticmethod
    def get_activity_emoji(activity: Activity) -> str:
        """Return emoji for display."""
        return {
            Activity.DIGGING: "⛏️",
            Activity.SWINGING: "🔄",
            Activity.DUMPING: "📤",
            Activity.WAITING: "⏸️",
            Activity.MOVING: "🚛",
            Activity.UNKNOWN: "❓",
        }.get(activity, "❓")

    @staticmethod
    def get_activity_color(activity: Activity) -> tuple[int, int, int]:
        """Return BGR color for visualization."""
        return {
            Activity.DIGGING: (0, 165, 255),    # Orange
            Activity.SWINGING: (255, 165, 0),   # Blue-orange
            Activity.DUMPING: (0, 255, 255),    # Yellow
            Activity.WAITING: (0, 0, 255),      # Red
            Activity.MOVING: (0, 255, 0),       # Green
            Activity.UNKNOWN: (128, 128, 128),  # Gray
        }.get(activity, (255, 255, 255))
