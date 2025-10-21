import os
import json
import time
import paho.mqtt.client as mqtt
import random

# Use the same MQTT broker as the main engine
MQTT_URL = os.getenv("MQTT_URL", "broker.hivemq.com")
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))

def main() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_URL, MQTT_PORT)
    print("Publisher connected to MQTT.")

    frame_count = 0
    while True:
        frame_count += 1
        payload = {
            "frame_id": f"sim_frame_{frame_count}",
            "detections": [
                {
                    "cls": "person", 
                    "conf": round(random.uniform(0.85, 0.99), 2), 
                    "xywh": [
                        random.randint(100, 400), 
                        random.randint(100, 300), 
                        random.randint(60, 100), 
                        random.randint(120, 180)
                    ]
                },
                {
                    "cls": "vehicle", 
                    "conf": round(random.uniform(0.75, 0.95), 2), 
                    "xywh": [
                        random.randint(500, 1000), 
                        random.randint(400, 600), 
                        random.randint(150, 250), 
                        random.randint(100, 150)
                    ]
                },
            ],
            "image_size": [1280, 720],
            "ts": time.time(),
        }
        client.publish("perception/ALPHA/detections", json.dumps(payload))
        print(f"Published fake detection for frame {frame_count}")
        time.sleep(5) 


if __name__ == "__main__":
    main()
