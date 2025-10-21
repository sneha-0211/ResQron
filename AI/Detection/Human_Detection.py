from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import paho.mqtt.client as mqtt
import json
import geocoder
import signal
import sys
import time
import traceback
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Setup MQTT client (configurable, with fallback and graceful handling)
MQTT_URL = os.getenv("MQTT_URL", "broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
client = mqtt.Client()
_mqtt_ok = False
try:
    client.connect(MQTT_URL, MQTT_PORT, 60)
    client.loop_start()
    _mqtt_ok = True
    print(f"MQTT connected to {MQTT_URL}:{MQTT_PORT}")
except Exception as e:
    print(f"MQTT connect failed: {e}. Proceeding without MQTT publish.")

# Webcam feed (or drone RTSP stream)
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # '0' for default cam or path/RTSP URL
cap_index = 0 if VIDEO_SOURCE == "0" else VIDEO_SOURCE
cap = cv2.VideoCapture(cap_index)
if not cap.isOpened():
    print(f"Could not open video source {VIDEO_SOURCE}. Set env VIDEO_SOURCE to a valid camera index or file/RTSP URL.")
    # Avoid hard crash; wait a bit and retry once
    time.sleep(1)
    cap = cv2.VideoCapture(cap_index)
    if not cap.isOpened():
        print("Exiting because no video source is available.")
        sys.exit(1)

# Create folder for detections
os.makedirs("detections", exist_ok=True)

# Graceful exit handler
def cleanup_and_exit(sig=None, frame=None):
    print("\nExiting detection script...")
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    try:
        if _mqtt_ok:
            client.loop_stop()
            client.disconnect()
    except Exception:
        pass
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

print("AI Detection Started... Press 'q' in window or Ctrl+C in terminal to exit.")

while True:
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Empty frame received. Retrying...")
            time.sleep(0.05)
            continue

        # Run detection
        results = model.predict(frame, conf=0.35, verbose=False)
        annotated = results[0].plot()

        alerts = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label in ["person", "fire", "car"] and conf > 0.35:
                alerts.append({"label": label, "confidence": conf})

        # Human detection logic
        if any(a["label"] == "person" for a in alerts):
            print("Human Detected! Sending Alert...")

            # Save annotated frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections/human_{timestamp}.jpg"
            try:
                cv2.imwrite(filename, annotated)
            except Exception as e:
                print(f"Failed to write image {filename}: {e}")

            # Get GPS location (simulated)
            try:
                g = geocoder.ip("me")
                lat, lng = g.latlng if g.latlng else (28.7041, 77.1025)
            except Exception:
                lat, lng = (28.7041, 77.1025)

            # Publish alert via MQTT (if available) and REST hook
            payload = {
                "event": "human_detected",
                "time": timestamp,
                "location": {"lat": lat, "lng": lng},
                "confidence": max(a["confidence"] for a in alerts if a["label"] == "person"),
                "image": filename
            }
            if _mqtt_ok:
                try:
                    client.publish("drone/DRONE_A/event", json.dumps(payload))
                except Exception as e:
                    print(f"MQTT publish failed (event): {e}")
            # Optional: call backend rescue alert webhook if configured
            try:
                import requests, os
                hook = os.getenv('RESCUE_ALERT_WEBHOOK')
                if hook:
                    requests.post(hook, json=payload, timeout=2)
            except Exception:
                pass

            # Optional: Auto cargo drop
            if _mqtt_ok:
                try:
                    client.publish("drone/DRONE_A/command", json.dumps({"cmd": "drop"}))
                except Exception as e:
                    print(f"MQTT publish failed (drop): {e}")

        # Show live annotated feed
        try:
            cv2.imshow("ResQron - Disaster AI Detection", annotated)
        except Exception:
            # If running headless, skip imshow
            pass

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cleanup_and_exit()

    except KeyboardInterrupt:
        cleanup_and_exit()
    except Exception as e:
        # Log error and attempt to continue
        print("Detection loop error:", e)
        traceback.print_exc()
        time.sleep(0.1)
