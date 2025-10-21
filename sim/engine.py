import time
import json
import random
import math
import os
import threading
import argparse
import paho.mqtt.client as mqtt
from .air_adapter import AirSimAdapter

# --- MQTT Connection Setup ---
raw_url = os.getenv('MQTT_URL', 'broker.hivemq.com')
if isinstance(raw_url, str) and raw_url.startswith('mqtt://'):
    raw_url = raw_url.replace('mqtt://', '', 1)
MQTT_URL = raw_url
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def connect_mqtt():
    """Establishes a connection to an MQTT broker with fallbacks."""
    for host in [MQTT_URL, 'broker.hivemq.com', 'test.mosquitto.org', 'localhost']:
        try:
            client.connect(host, MQTT_PORT, 60)
            client.loop_start()
            print(f"MQTT connected to {host}:{MQTT_PORT}")
            return True
        except Exception as e:
            print(f"MQTT connect failed to {host}:{MQTT_PORT} -> {e}")
    return False

if not connect_mqtt():
    raise SystemExit('Unable to connect to any MQTT broker. Set MQTT_URL/MQTT_PORT correctly and ensure network access.')

# --- Pure Simulation Vehicle (for running without AirSim) ---
class SimVehicle:
    def __init__(self, callsign, home=(28.6, 77.2), speed=5.0):
        self.callsign = callsign
        self.home = list(home)
        self.pos = list(home)
        self.speed = speed
        self.battery = 100.0
        self.path = []
        self.mode = 'IDLE'
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.running = False
        print(f"Initialized SimVehicle: {callsign}")

    def publish(self):
        topic = f"drone/{self.callsign}/telemetry"
        payload = {
            "callsign": self.callsign,
            "lat": round(self.pos[0], 6),
            "lng": round(self.pos[1], 6),
            "alt": 100.0,
            "battery": round(self.battery, 1),
            "mode": self.mode,
            "speed": self.speed
        }
        client.publish(topic, json.dumps(payload))

    def set_path(self, path):
        self.path = path
        if self.path:
            self.mode = 'GUIDED'

    def run(self):
        self.running = True
        while self.running:
            dt = 0.5
            self.battery -= dt * 0.02

            if self.battery <= 20 and self.mode != 'RTL':
                self.mode = 'RTL'
                self.path = [self.home[:]]

            if self.path:
                tgt = self.path[0]
                dx = tgt[0] - self.pos[0]
                dy = tgt[1] - self.pos[1]
                dist = math.hypot(dx, dy)
                if dist < 0.0001:
                    self.path.pop(0)
                    if not self.path:
                        self.mode = 'IDLE' if self.mode != 'RTL' else 'LAND'
                else:
                    step = 0.00005 * self.speed
                    self.pos[0] += (dx / dist) * step
                    self.pos[1] += (dy / dist) * step
            
            self.pos[0] += random.uniform(-1e-6, 1e-6)
            self.pos[1] += random.uniform(-1e-6, 1e-6)
            self.publish()
            time.sleep(dt)
            
    def start(self):
        if not self.running:
            self.thread.start()

# --- Main Engine Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run the Drone Simulation Engine.")
    parser.add_argument('--sim-only', action='store_true', help="Run in pure simulation mode without connecting to AirSim.")
    args = parser.parse_args()

    if args.sim_only:
        print("Running in PURE SIMULATION mode.")
        # --- MQTT Callbacks for SimVehicle ---
        vehicles = {
            'ALPHA': SimVehicle('ALPHA', home=(28.600, 77.200)),
            'BETA': SimVehicle('BETA', home=(28.605, 77.205))
        }
        for v in vehicles.values():
            v.start()

        def on_mission_assign(client, userdata, msg):
            try:
                topic_parts = msg.topic.split('/')
                callsign = topic_parts[1]
                mission = json.loads(msg.payload.decode())
                if callsign in vehicles and 'waypoints' in mission:
                    # Simple path: fly to each waypoint and then RTL
                    path = [wp for wp in mission['waypoints']]
                    vehicles[callsign].set_path(path)
                    print(f"Mission assigned to SimVehicle {callsign}")
            except Exception as e:
                print(f"Error processing mission assignment: {e}")

        client.message_callback_add('mission/+/assign', on_mission_assign)
        client.subscribe('mission/+/assign')
        
        while True:
            time.sleep(1)

    else:
        print("Running in AIRSIM INTEGRATION mode.")
        airsim_adapter = AirSimAdapter(client, vehicle_name="Drone1", callsign="ALPHA")
        airsim_adapter.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down AirSim adapter.")
            airsim_adapter.stop()

if __name__ == "__main__":
    main()
