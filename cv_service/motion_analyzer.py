"""
motion_analyzer.py — Optical Flow based motion analysis
=======================================================
Determines if equipment is ACTIVE or INACTIVE using dense optical flow.

Key insight for articulated equipment (excavators):
    - The arm (boom) moves while the tracks are stationary → still ACTIVE!
    - We split the bounding box into regions and analyze each separately.
    - motion_source: "arm_only" | "full_body" | "none"
"""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass


class MotionState(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


@dataclass
class MotionResult:
    state: MotionState
    motion_source: str        # "arm_only" | "full_body" | "none"
    upper_flow_mag: float     # Mean flow magnitude in upper region (arm)
    lower_flow_mag: float     # Mean flow magnitude in lower region (body/tracks)
    overall_flow_mag: float   # Overall mean flow magnitude
    flow_dx: float            # Mean horizontal flow (for activity classification)
    flow_dy: float            # Mean vertical flow (for activity classification)


class MotionAnalyzer:
    """
    Analyzes equipment motion using dense Farneback optical flow.
    
    Region-based analysis:
        ┌──────────────────┐
        │  UPPER REGION    │  → arm / boom (40% of bbox height)
        │  (arm/boom)      │
        ├──────────────────┤
        │  LOWER REGION    │  → tracks / body (60% of bbox height)
        │  (tracks/body)   │
        └──────────────────┘
    
    Decision logic:
        upper_flow > ARM_THRESHOLD   → ACTIVE (arm_only or full_body)
        lower_flow > BODY_THRESHOLD  → ACTIVE (full_body)
        both < threshold             → INACTIVE
    """

    def __init__(
        self,
        arm_threshold: float = 1.5,    # Min flow magnitude to consider arm moving
        body_threshold: float = 1.0,   # Min flow magnitude to consider body moving
        upper_ratio: float = 0.45,     # Fraction of bbox height for upper region
    ):
        self.arm_threshold = arm_threshold
        self.body_threshold = body_threshold
        self.upper_ratio = upper_ratio

        # Farneback optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Previous frame (grayscale) cache per track_id
        self._prev_gray: dict[int, np.ndarray] = {}
        self._prev_bbox: dict[int, list[int]] = {}

    def analyze(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: list[int],
    ) -> MotionResult:
        """
        Analyze motion for a single tracked equipment.

        Args:
            frame:    Current frame (BGR)
            track_id: Equipment track ID (for per-track history)
            bbox:     [x1, y1, x2, y2]

        Returns:
            MotionResult with state and flow metrics
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if x2 <= x1 or y2 <= y1:
            return MotionResult(MotionState.INACTIVE, "none", 0, 0, 0, 0, 0)

        # Current crop (grayscale)
        crop = frame[y1:y2, x1:x2]
        curr_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (128, 96))  # normalize size

        # Initialize on first frame for this track
        if track_id not in self._prev_gray:
            self._prev_gray[track_id] = curr_gray
            return MotionResult(MotionState.INACTIVE, "none", 0, 0, 0, 0, 0)

        prev_gray = self._prev_gray[track_id]

        # Handle shape mismatch (bbox changed a lot)
        if prev_gray.shape != curr_gray.shape:
            self._prev_gray[track_id] = curr_gray
            return MotionResult(MotionState.INACTIVE, "none", 0, 0, 0, 0, 0)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self.flow_params
        )

        # Update history
        self._prev_gray[track_id] = curr_gray

        # Compute flow magnitude & direction
        flow_h, flow_w = flow.shape[:2]
        split_y = int(flow_h * self.upper_ratio)

        upper_flow = flow[:split_y, :, :]  # arm/boom region
        lower_flow = flow[split_y:, :, :]  # tracks/body region

        def mean_magnitude(f):
            mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
            return float(np.mean(mag))

        upper_mag = mean_magnitude(upper_flow)
        lower_mag = mean_magnitude(lower_flow)
        overall_mag = mean_magnitude(flow)

        # Mean flow direction (dx, dy) for activity classification
        flow_dx = float(np.mean(flow[..., 0]))
        flow_dy = float(np.mean(flow[..., 1]))

        # Determine motion state
        upper_active = upper_mag > self.arm_threshold
        lower_active = lower_mag > self.body_threshold

        if upper_active and lower_active:
            state = MotionState.ACTIVE
            motion_source = "full_body"
        elif upper_active:
            state = MotionState.ACTIVE
            motion_source = "arm_only"    # Articulated motion detected!
        elif lower_active:
            state = MotionState.ACTIVE
            motion_source = "body_only"
        else:
            state = MotionState.INACTIVE
            motion_source = "none"

        return MotionResult(
            state=state,
            motion_source=motion_source,
            upper_flow_mag=upper_mag,
            lower_flow_mag=lower_mag,
            overall_flow_mag=overall_mag,
            flow_dx=flow_dx,
            flow_dy=flow_dy,
        )

    def remove_track(self, track_id: int):
        """Clean up when a track is permanently removed."""
        self._prev_gray.pop(track_id, None)
        self._prev_bbox.pop(track_id, None)

    def visualize_flow(
        self, frame: np.ndarray, bbox: list[int], motion: MotionResult
    ) -> np.ndarray:
        """Draw motion analysis overlay on frame."""
        vis = frame.copy()
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        split_y = int(bh * self.upper_ratio)

        # Upper region (arm)
        arm_y = y1 + split_y
        color_upper = (0, 255, 0) if motion.upper_flow_mag > self.arm_threshold else (0, 0, 128)
        color_lower = (0, 255, 0) if motion.lower_flow_mag > self.body_threshold else (0, 0, 128)

        cv2.rectangle(vis, (x1, y1), (x2, arm_y), color_upper, 1)
        cv2.rectangle(vis, (x1, arm_y), (x2, y2), color_lower, 1)

        # Flow magnitude text
        cv2.putText(vis, f"A:{motion.upper_flow_mag:.1f}", (x1+2, y1+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_upper, 1)
        cv2.putText(vis, f"B:{motion.lower_flow_mag:.1f}", (x1+2, arm_y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_lower, 1)

        return vis
