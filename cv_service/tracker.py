"""
tracker.py — Object Tracking Wrapper (using ByteTrack via Ultralytics)
======================================================================
"""

from ultralytics.trackers import BOTSORT, BYTETracker
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

class SimpleTracker:
    """
    Since Ultralytics handles tracking internally if you call `model.track()`, 
    we will just wrap the Ultralytics tracker to maintain state manually 
    or just use the ultralytics tracker output directly.
    """
    pass
