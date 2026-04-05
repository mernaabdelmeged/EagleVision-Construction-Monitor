import json
import time
import psycopg2
from kafka import KafkaConsumer
from datetime import datetime

def connect_db():
    while True:
        try:
            conn = psycopg2.connect(
                host="timescaledb",
                database="eaglevision_db",
                user="eaglevision",
                password="password"
            )
            return conn
        except Exception as e:
            print(f"Waiting for database... {e}")
            time.sleep(5)

def main():
    print("Starting DB Writer Service...")
    conn = connect_db()
    cursor = conn.cursor()

    consumer = KafkaConsumer(
        'equipment-events',
        bootstrap_servers=['kafka:29092'],
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    for message in consumer:
        data = message.value
        print(f"Logging data for {data['equipment_id']} at {data['timestamp']}")
        
        try:
            cursor.execute(
                """
                INSERT INTO utilization_logs (
                    time, equipment_id, equipment_class, current_state, 
                    current_activity, total_active_seconds, 
                    total_idle_seconds, utilization_percent
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    datetime.now(),
                    data['equipment_id'],
                    data['equipment_class'],
                    data['utilization']['current_state'],
                    data['utilization']['current_activity'],
                    data['time_analytics']['total_active_seconds'],
                    data['time_analytics']['total_idle_seconds'],
                    data['time_analytics']['utilization_percent']
                )
            )
            conn.commit()
        except Exception as e:
            print(f"Database error: {e}")
            conn.rollback()

if __name__ == "__main__":
    main()
