import streamlit as st
import pandas as pd
import json
from kafka import KafkaConsumer
import threading
import time

# --- Page Config ---
st.set_page_config(
    page_title="EagleVision Equipment Monitor",
    page_icon="🚧",
    layout="wide"
)

# --- Force Zero-Start Cleanup ---
import os
import json
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(root_dir, 'data', 'live_snapshot.json')
    if os.path.exists(json_path):
        os.remove(json_path)
except:
    pass

# --- Robust Data Bridge (Streamlit & Thread-safe) ---
class DataBridge:
    def __init__(self):
        self.data = {}
    def update(self, new_data):
        self.data.update(new_data)
    def set(self, fresh_snapshot):
        self.data = fresh_snapshot
    def get_all(self):
        return self.data.copy()

@st.cache_resource
def get_bridge():
    return DataBridge()

bridge = get_bridge()

def kafka_listener(bridge_obj):
    """Background thread to consume Kafka messages"""
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            'equipment-events',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            consumer_timeout_ms=1000
        )
        for message in consumer:
            data = message.value
            eq_id = data.get("equipment_id")
            if eq_id:
                bridge_obj.update({eq_id: data})
    except:
        pass

def local_json_listener(bridge_obj):
    """Fallback: Read from local JSON snapshot if Kafka is slow"""
    import os
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'live_snapshot.json')
    while True:
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    snapshot = json.load(f)
                    valid_snapshot = {k: v for k, v in snapshot.items() if isinstance(v, dict)}
                    if valid_snapshot:
                        bridge_obj.set(valid_snapshot)
            else:
                # If file is deleted, clear the UI
                bridge_obj.set({})
        except:
            pass
        time.sleep(0.1)

# Start background listeners once
if 'listener_started' not in st.session_state:
    st.session_state.listener_started = True
    threading.Thread(target=kafka_listener, args=(bridge,), daemon=True).start()
    threading.Thread(target=local_json_listener, args=(bridge,), daemon=True).start()

# Sync current data
current_data = bridge.get_all()

# --- Extra Direct-Read Layer ---
# If bridge is empty, try a quick direct read to be safe
if not current_data:
    try:
        import os
        # Absolute path discovery
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(root_dir, 'data', 'live_snapshot.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                current_data = json.load(f)
                bridge.set(current_data)
    except:
        pass

st.session_state.equipment_data = current_data

# --- UI Layout ---
st.title("🚧 EagleVision: Equipment Utilization Dashboard")
st.markdown("Real-time monitoring of construction equipment dwell time and activities.")

# Sidebar Controls
with st.sidebar:
    st.header("🛠️ Dashboard Controls")
    if st.button("🗑️ Reset All Statistics"):
        bridge.set({})
        st.session_state.equipment_data = {}
        st.success("Dashboard Reset!")
        time.sleep(0.5)
        st.rerun()
    
    st.info("💡 Tip: Use this to clear old data before starting a new Recording.")

# Filter valid data
valid_data = {k: v for k, v in st.session_state.equipment_data.items() if isinstance(v, dict)}

# Metrics row
col1, col2, col3 = st.columns(3)
machines_tracked = len(valid_data)
active_machines = sum(1 for d in valid_data.values() if d.get('utilization', {}).get('current_state') == "ACTIVE")
idle_machines = machines_tracked - active_machines

col1.metric("Machines Tracked", machines_tracked)
col2.metric("🟢 Active (Working)", active_machines)
col3.metric("🔴 Idle (Waiting)", idle_machines)

st.markdown("---")

# Dynamic Machine Cards
if not st.session_state.equipment_data:
    st.info("Waiting for data from CV Pipeline... (Ensure Kafka and CV Service are running)")
else:
    for eq_id, data in st.session_state.equipment_data.items():
        state = data.get("utilization", {}).get("current_state", "UNKNOWN")
        activity = data.get("utilization", {}).get("current_activity", "UNKNOWN")
        dwell = data.get("dwell_time", {})
        analytics = data.get("time_analytics", {})
        
        is_reid = "🔄 Re-identified" if data.get("is_reidentified", False) else ""
        
        # Determine styling based on state
        card_color = "#e6ffe6" if state == "ACTIVE" else "#ffe6e6"
        border_color = "green" if state == "ACTIVE" else "red"
        
        with st.container():
            st.markdown(f"""
            <div style="background-color: {card_color}; padding: 15px; border-left: 5px solid {border_color}; border-radius: 5px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin:0;">{eq_id} ({data.get('equipment_class', 'unknown').capitalize()})</h3>
                    <span style="color: gray; font-size: 0.8em;">{data.get('timestamp', '')}</span>
                </div>
                <p style="margin: 5px 0;"><strong>Status:</strong> {state} | <strong>Activity:</strong> {activity} | <strong>Motion:</strong> {data.get('utilization', {}).get('motion_source', 'N/A')}</p>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div>
                        <p style="margin:0; color: gray; font-size: 0.9em;">Total Working Time</p>
                        <h4 style="margin:0; color: #2ecc71;">{analytics.get('total_active_seconds', 0):.1f} s</h4>
                    </div>
                    <div>
                        <p style="margin:0; color: gray; font-size: 0.9em;">Total Idle Time</p>
                        <h4 style="margin:0; color: #e74c3c;">{analytics.get('total_idle_seconds', 0):.1f} s</h4>
                    </div>
                    <div>
                        <p style="margin:0; color: gray; font-size: 0.9em;">Total Tracked</p>
                        <h4 style="margin:0;">{analytics.get('total_tracked_seconds', 0):.1f} s</h4>
                    </div>
                    <div>
                        <p style="margin:0; color: gray; font-size: 0.9em;">Utilization %</p>
                        <h4 style="margin:0; color: #3498db;">{analytics.get('utilization_percent', 0):.1f}%</h4>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Auto-refresh mechanism
time.sleep(1)
st.rerun()
