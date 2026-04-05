"""
reid_module.py — Visual Re-Identification for construction equipment
=================================================================
Problem: ByteTrack loses track ID when equipment disappears behind occlusion.
         New ID = dwell time resets to ZERO ❌

Solution: Maintain a gallery of visual embeddings per track ID.
          When a new track appears, compare embeddings to lost tracks gallery.
          If similarity > threshold → restore old ID + accumulated dwell time ✅

Method: ResNet-based feature extractor → 512-dim embedding → Cosine similarity
"""

import cv2
import numpy as np
from collections import defaultdict
import time


class SimpleEmbeddingExtractor:
    """
    Lightweight visual embedding extractor using pixel statistics + HOG-like features.
    No extra model needed — works purely with OpenCV.
    
    For production: replace with OSNet (torchreid) or a fine-tuned ResNet.
    """

    def __init__(self, output_dim: int = 256):
        self.output_dim = output_dim

    def extract(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        """
        Extract visual embedding from bounding box region.

        Args:
            frame: Full image (BGR)
            bbox: [x1, y1, x2, y2]

        Returns:
            L2-normalized embedding vector of shape (output_dim,)
        """
        x1, y1, x2, y2 = bbox
        # Clamp to frame boundaries
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.output_dim)

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (64, 128))  # Standard Re-ID input size

        features = []

        # --- Feature 1: Color histogram (HSV) ---
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.append(hist)  # 3 * 32 = 96 dims

        # --- Feature 2: LBP-like texture (upper vs lower body) ---
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        upper = gray[:64, :]   # top half (arm/boom)
        lower = gray[64:, :]   # bottom half (tracks/body)

        for region in [upper, lower]:
            # Simple gradient features
            gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            ang = np.arctan2(gy, gx)

            # Histogram of gradients (8 bins)
            mag_hist, _ = np.histogram(mag, bins=16, range=(0, 500))
            ang_hist, _ = np.histogram(ang, bins=16, range=(-np.pi, np.pi))
            mag_hist = mag_hist / (mag_hist.sum() + 1e-6)
            ang_hist = ang_hist / (ang_hist.sum() + 1e-6)
            features.extend([mag_hist, ang_hist])  # 2 * 2 * 16 = 64 dims

        # --- Feature 3: Average color per stripe ---
        stripes = np.array_split(crop, 8, axis=0)
        for stripe in stripes:
            mean_color = stripe.mean(axis=(0, 1)) / 255.0  # 3 dims per stripe
            features.append(mean_color)  # 8 * 3 = 24 dims

        # Concatenate all features
        embedding = np.concatenate([f.flatten() for f in features])

        # Pad or truncate to output_dim
        if len(embedding) < self.output_dim:
            embedding = np.pad(embedding, (0, self.output_dim - len(embedding)))
        else:
            embedding = embedding[:self.output_dim]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)


class ReIDModule:
    """
    Equipment Re-Identification module.
    
    Maintains:
        - active_gallery: {track_id → embedding} for currently visible equipment
        - lost_gallery:   {track_id → {embedding, dwell_state}} for recently lost tracks
    
    When new track appears:
        1. Compare embedding to lost_gallery
        2. If best cosine similarity > threshold → re-use old track_id
        3. Restore accumulated dwell time from old track
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_lost_frames: int = 150,   # ~5 sec at 30fps
        max_lost_tracks: int = 50,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_lost_frames = max_lost_frames
        self.max_lost_tracks = max_lost_tracks

        self.extractor = SimpleEmbeddingExtractor(output_dim=256)

        # Active tracks: {tracker_id → embedding}
        self.active_gallery: dict[int, np.ndarray] = {}
        # Store last known bbox: {tracker_id → bbox}
        self.active_bboxes: dict[int, list[int]] = {}

        # Lost tracks: {old_tracker_id → {embedding, frame_count, dwell_state, last_bbox}}
        self.lost_gallery: dict[int, dict] = {}

        # ID remapping: {new_tracker_id → old_tracker_id}
        self.id_remap: dict[int, int] = {}

        self.frame_count = 0

    def update_active(self, frame: np.ndarray, tracker_id: int, bbox: list[int]):
        """Update embedding for an actively tracked equipment."""
        self.active_bboxes[tracker_id] = bbox
        
        embedding = self.extractor.extract(frame, bbox)
        # Exponential moving average for smoother embeddings
        if tracker_id in self.active_gallery:
            alpha = 0.7
            self.active_gallery[tracker_id] = (
                alpha * self.active_gallery[tracker_id] + (1 - alpha) * embedding
            )
            # Re-normalize
            norm = np.linalg.norm(self.active_gallery[tracker_id])
            if norm > 0:
                self.active_gallery[tracker_id] /= norm
        else:
            self.active_gallery[tracker_id] = embedding

    def mark_lost(self, tracker_id: int, dwell_state: dict = None):
        """
        Called when tracker loses a track.
        Moves embedding from active → lost gallery.
        """
        if tracker_id in self.active_gallery:
            self.lost_gallery[tracker_id] = {
                "embedding": self.active_gallery.pop(tracker_id),
                "last_bbox": self.active_bboxes.pop(tracker_id, [0, 0, 0, 0]),
                "frames_lost": 0,
                "dwell_state": dwell_state or {},
                "lost_at_frame": self.frame_count,
            }

    def try_reidentify(
        self, frame: np.ndarray, new_tracker_id: int, bbox: list[int], class_name: str
    ) -> tuple[int, bool, dict]:
        """
        Try to match a new track to a lost track.
        Fallback to Spatial distance if visual match is poor.
        """
        new_embedding = self.extractor.extract(frame, bbox)
        
        # Center of new bbox
        cx_new = (bbox[0] + bbox[2]) / 2.0
        cy_new = (bbox[1] + bbox[3]) / 2.0

        best_match_id = None
        best_similarity = 0.0
        
        # Check spatial overlap first (for perfectly stationary cameras and equipment)
        for old_id, lost_data in self.lost_gallery.items():
            last_bbox = lost_data["last_bbox"]
            cx_old = (last_bbox[0] + last_bbox[2]) / 2.0
            cy_old = (last_bbox[1] + last_bbox[3]) / 2.0
            
            # Pixel distance between centers
            dist = ((cx_new - cx_old)**2 + (cy_new - cy_old)**2) ** 0.5
            
            # If the box appeared in almost the EXACT same spot (< 50 pixels shift)
            if dist < 50.0:
                best_match_id = old_id
                best_similarity = 1.0  # Force it to match!
                break

        # If spatial fallback didn't catch it, rely on visual embeddings
        if best_match_id is None:
            for old_id, lost_data in self.lost_gallery.items():
                old_embedding = lost_data["embedding"]
                
                # Cosine similarity
                sim = float(np.dot(new_embedding, old_embedding))
    
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id = old_id

        if best_match_id is not None and best_similarity >= self.similarity_threshold:
            # Re-identified! Use old ID, restore dwell state
            restored_state = self.lost_gallery[best_match_id]["dwell_state"]
            del self.lost_gallery[best_match_id]

            # Register remapping
            self.id_remap[new_tracker_id] = best_match_id
            self.active_gallery[best_match_id] = new_embedding
            self.active_bboxes[best_match_id] = bbox

            print(f"[ReID] ✅ Re-identified track {new_tracker_id} → {best_match_id} "
                  f"(similarity/spatial_match={best_similarity:.3f})")
            return best_match_id, True, restored_state
        else:
            # New equipment — assign new ID
            self.active_gallery[new_tracker_id] = new_embedding
            self.active_bboxes[new_tracker_id] = bbox
            return new_tracker_id, False, {}

    def resolve_id(self, tracker_id: int) -> int:
        """Get the canonical ID (after any remapping)."""
        return self.id_remap.get(tracker_id, tracker_id)

    def tick(self):
        """
        Called every frame. Increments lost-frame counters.
        Removes tracks that have been lost too long.
        """
        self.frame_count += 1
        expired = []
        for old_id, lost_data in self.lost_gallery.items():
            lost_data["frames_lost"] += 1
            if lost_data["frames_lost"] > self.max_lost_frames:
                expired.append(old_id)
        for old_id in expired:
            print(f"[ReID] ❌ Track {old_id} expired from lost gallery")
            del self.lost_gallery[old_id]

    def get_stats(self) -> dict:
        return {
            "active_tracks": len(self.active_gallery),
            "lost_tracks": len(self.lost_gallery),
            "remappings": len(self.id_remap),
        }
