"""
kafka_producer.py — Sends events to Apache Kafka
================================================
"""

import json
from kafka import KafkaProducer
from typing import Dict, Any

class EquipmentEventProducer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='equipment-events'):
        self.topic = topic
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=3
            )
            self.connected = True
            print(f"[KafkaProducer] Connected to {bootstrap_servers}")
        except Exception as e:
            print(f"[KafkaProducer] Failed to connect: {e}")
            self.connected = False

    def send_event(self, event: Dict[str, Any]):
        if not self.connected:
            return False
            
        try:
            self.producer.send(self.topic, event)
            return True
        except Exception as e:
            print(f"[KafkaProducer] Error sending event: {e}")
            return False
            
    def close(self):
        if self.connected:
            self.producer.flush()
            self.producer.close()
