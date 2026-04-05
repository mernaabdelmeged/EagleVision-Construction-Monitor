"""
dwell_time_tracker.py — Equipment Dwell (Idle) Time Tracking
=============================================================
🔴 MOST CRITICAL MODULE of the system

Tracks idle time per equipment instance with:
    1. Current session idle time  (how long idle NOW in this continuous stop)
    2. Total accumulated idle time (sum of ALL idle sessions since first seen)
    3. Full session history        (list of all idle episodes with start/end/duration)

This data persists across Re-ID events — the Re-ID module restores the state
when an equipment reappears after occlusion.

Business context:
    Construction equipment is rented per 8-hour shift.
    Every idle minute = wasted money.
    Alert when equipment is idle > threshold (default: 5 minutes).
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IdleSession:
    """Represents a single continuous idle period."""
    session_id: int
    start_time: float       # Unix timestamp
    end_time: Optional[float] = None
    duration_sec: float = 0.0
    is_active: bool = True  # True = ongoing, False = completed

    def finalize(self, end_time: float):
        self.end_time = end_time
        self.duration_sec = end_time - self.start_time
        self.is_active = False


@dataclass
class MachineState:
    """Complete state for one tracked equipment instance."""
    equipment_id: str
    equipment_class: str

    # Current state
    current_state: str = "INACTIVE"       # "ACTIVE" | "INACTIVE"
    current_activity: str = "WAITING"     # "DIGGING" | "SWINGING" | "DUMPING" | "WAITING" | "MOVING"
    motion_source: str = "none"

    # Timing
    first_seen_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    state_start_time: float = field(default_factory=time.time)  # When current state began

    # Accumulated totals
    total_tracked_sec: float = 0.0
    total_active_sec: float = 0.0
    total_idle_sec: float = 0.0

    # Current idle session
    current_session_idle_sec: float = 0.0  # Duration of ongoing idle (if INACTIVE)
    idle_session_count: int = 0

    # Session history
    idle_sessions: list = field(default_factory=list)  # List of IdleSession
    _current_idle_session: Optional[object] = field(default=None, repr=False)

    # Re-ID flag
    is_reidentified: bool = False
    reidentified_count: int = 0

    @property
    def utilization_percent(self) -> float:
        if self.total_tracked_sec <= 0:
            return 0.0
        return round(self.total_active_sec / self.total_tracked_sec * 100, 1)

    @property
    def idle_percent(self) -> float:
        return round(100.0 - self.utilization_percent, 1)


class DwellTimeTracker:
    """
    Tracks idle/active time for all detected equipment.
    
    Updates state machine per track per frame:
    
        ACTIVE ──(stops moving)──► INACTIVE
                                       │ accumulate idle time
        ACTIVE ◄──(starts moving)── INACTIVE
    
    Re-ID integration:
        When Re-ID restores an old track, call restore_state() to
        merge the previous accumulated data into the new track.
    """

    IDLE_ALERT_THRESHOLD_SEC = 300  # 5 minutes — alert when exceeded

    def __init__(self):
        self.machines: dict[str, MachineState] = {}

    def update(
        self,
        equipment_id: str,
        equipment_class: str,
        current_state: str,      # "ACTIVE" | "INACTIVE"
        current_activity: str,
        motion_source: str,
        timestamp: float,        # Current video timestamp in seconds
        is_reidentified: bool = False,
    ) -> MachineState:
        """
        Update dwell time state for one equipment.
        Called every processed frame.

        Returns: Updated MachineState
        """
        now = timestamp

        # Initialize if new equipment
        if equipment_id not in self.machines:
            machine = MachineState(
                equipment_id=equipment_id,
                equipment_class=equipment_class,
                current_state=current_state,
                current_activity=current_activity,
                motion_source=motion_source,
                first_seen_time=now,
                last_update_time=now,
                state_start_time=now,
            )
            self.machines[equipment_id] = machine

            # Start idle session immediately if INACTIVE
            if current_state == "INACTIVE":
                self._start_idle_session(machine, now)
        else:
            machine = self.machines[equipment_id]

        # --- Time accounting ---
        dt = max(0.0, now - machine.last_update_time)  # seconds since last update
        machine.total_tracked_sec += dt

        if machine.current_state == "ACTIVE":
            machine.total_active_sec += dt
        else:
            machine.total_idle_sec += dt
            # Update current session idle time
            if machine._current_idle_session is not None:
                machine.current_session_idle_sec = now - machine.state_start_time

        # --- State transition ---
        if current_state != machine.current_state:
            self._handle_state_transition(machine, current_state, now)

        # --- Update current fields ---
        machine.current_state = current_state
        machine.current_activity = current_activity
        machine.motion_source = motion_source
        machine.last_update_time = now

        if is_reidentified and not machine.is_reidentified:
            machine.is_reidentified = True
            machine.reidentified_count += 1

        return machine

    def _handle_state_transition(
        self, machine: MachineState, new_state: str, now: float
    ):
        """Handle ACTIVE↔INACTIVE transitions."""
        if machine.current_state == "INACTIVE" and new_state == "ACTIVE":
            # Equipment started moving — close current idle session
            if machine._current_idle_session is not None:
                session = machine._current_idle_session
                session.finalize(now)
                machine.idle_sessions.append(session)
                machine._current_idle_session = None
                machine.current_session_idle_sec = 0.0

        elif machine.current_state == "ACTIVE" and new_state == "INACTIVE":
            # Equipment stopped — start new idle session
            self._start_idle_session(machine, now)

        machine.state_start_time = now

    def _start_idle_session(self, machine: MachineState, now: float):
        """Create a new idle session."""
        machine.idle_session_count += 1
        machine._current_idle_session = IdleSession(
            session_id=machine.idle_session_count,
            start_time=now,
        )
        machine.current_session_idle_sec = 0.0

    def restore_state(self, new_equipment_id: str, old_state: dict):
        """
        Called by Re-ID module when restoring a previously lost track.
        Merges old accumulated data into new track entry.
        
        Args:
            new_equipment_id: The new track's equipment_id
            old_state: Dict with previous MachineState data
        """
        if new_equipment_id not in self.machines:
            return
        machine = self.machines[new_equipment_id]

        # Restore accumulated totals
        machine.total_tracked_sec += old_state.get("total_tracked_sec", 0)
        machine.total_active_sec += old_state.get("total_active_sec", 0)
        machine.total_idle_sec += old_state.get("total_idle_sec", 0)
        machine.idle_session_count += old_state.get("idle_session_count", 0)

        # Restore session history
        old_sessions = old_state.get("idle_sessions", [])
        machine.idle_sessions = old_sessions + machine.idle_sessions

        machine.is_reidentified = True
        machine.reidentified_count = old_state.get("reidentified_count", 0) + 1

        print(f"[DwellTime] ✅ Restored state for {new_equipment_id}: "
              f"total_idle={machine.total_idle_sec:.1f}s")

    def get_machine_state(self, equipment_id: str) -> Optional[MachineState]:
        return self.machines.get(equipment_id)

    def get_all_machines(self) -> dict[str, MachineState]:
        return self.machines.copy()

    def export_state(self, equipment_id: str) -> dict:
        """Export machine state as dict (for Re-ID restoration)."""
        machine = self.machines.get(equipment_id)
        if not machine:
            return {}
        return {
            "total_tracked_sec": machine.total_tracked_sec,
            "total_active_sec": machine.total_active_sec,
            "total_idle_sec": machine.total_idle_sec,
            "idle_session_count": machine.idle_session_count,
            "idle_sessions": machine.idle_sessions,
            "reidentified_count": machine.reidentified_count,
        }

    def is_idle_alert(self, equipment_id: str) -> bool:
        """Return True if equipment has been idle longer than threshold."""
        machine = self.machines.get(equipment_id)
        if not machine:
            return False
        return (
            machine.current_state == "INACTIVE"
            and machine.current_session_idle_sec >= self.IDLE_ALERT_THRESHOLD_SEC
        )

    @staticmethod
    def format_seconds(sec: float) -> str:
        """Format seconds → 'MM:SS'."""
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
