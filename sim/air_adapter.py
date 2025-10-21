import os
import json
import time
import threading
import paho.mqtt.client as mqtt
import airsim

class AirSimAdapter:
    """
    Connects to an AirSim instance, controls a drone, and bridges telemetry
    and commands with an MQTT broker.
    """
    def __init__(self, mqtt_client: mqtt.Client, vehicle_name: str, callsign: str):
        self.mqtt_client = mqtt_client
        self.vehicle_name = vehicle_name
        self.callsign = callsign
        
        print("Connecting to AirSim...")
        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        self.airsim_client.enableApiControl(True, self.vehicle_name)
        self.airsim_client.armDisarm(True, self.vehicle_name)
        print("AirSim connection successful and API control enabled.")

        self.running = False
        self.thread = threading.Thread(target=self._run_telemetry_loop)
        self.thread.daemon = True

    def start(self):
        """Starts the telemetry and mission handling loop."""
        if not self.running:
            self.running = True
            self.thread.start()
            self._setup_mission_subscription()
            print(f"AirSim adapter for {self.callsign} started.")

    def stop(self):
        """Stops the adapter and releases AirSim control."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.airsim_client.armDisarm(False, self.vehicle_name)
        self.airsim_client.enableApiControl(False, self.vehicle_name)
        print(f"AirSim adapter for {self.callsign} stopped.")

    def _setup_mission_subscription(self):
        """Subscribes to mission assignment topics for this specific drone."""
        topic = f"mission/{self.callsign}/assign"
        self.mqtt_client.subscribe(topic)
        self.mqtt_client.message_callback_add(topic, self._on_mission_assign)
        print(f"Subscribed to mission topic: {topic}")

    def _on_mission_assign(self, client, userdata, msg):
        """Callback function to handle incoming mission assignments."""
        try:
            mission = json.loads(msg.payload.decode())
            print(f"\nReceived mission for {self.callsign}: {mission.get('_id', 'N/A')}")
            
            if 'waypoints' in mission and mission['waypoints']:
                waypoints = mission['waypoints']
                # Start mission execution in a new thread to not block telemetry
                mission_thread = threading.Thread(target=self._execute_mission, args=(waypoints,))
                mission_thread.start()
            else:
                print("Mission received with no waypoints.")
                
        except Exception as e:
            print(f"Error processing mission payload: {e}")

    def _execute_mission(self, waypoints: list):
        """Commands the AirSim drone to fly a path defined by waypoints."""
        print(f"{self.callsign} taking off...")
        self.airsim_client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.airsim_client.moveToZAsync(-20, 5, vehicle_name=self.vehicle_name).join()

        # Convert waypoints to AirSim's Vector3r format and fly path
        airsim_path = [airsim.Vector3r(wp['lat'], wp['lng'], -20) for wp in waypoints]
        
        if not airsim_path:
            print("No valid waypoints to fly.")
            return

        print(f"{self.callsign} flying mission path with {len(airsim_path)} waypoints...")
        self.airsim_client.moveOnPathAsync(
            airsim_path,
            velocity=10,
            timeout_sec=3600,
            vehicle_name=self.vehicle_name
        ).join()

        print(f"Mission for {self.callsign} complete. Returning to land.")
        self.airsim_client.landAsync(vehicle_name=self.vehicle_name).join()
        print(f"{self.callsign} has landed.")
        
    def _run_telemetry_loop(self):
        """Continuously fetches and publishes drone telemetry."""
        while self.running:
            try:
                # Get state from AirSim
                state = self.airsim_client.getMultirotorState(vehicle_name=self.vehicle_name)
                pos = state.gps_location
                battery_info = self.airsim_client.getEnergyInfo(vehicle_name=self.vehicle_name)
                
                # Prepare payload
                payload = {
                    "callsign": self.callsign,
                    "lat": round(pos.latitude, 6),
                    "lng": round(pos.longitude, 6),
                    "alt": round(pos.altitude, 2),
                    "battery": round(battery_info.battery_level * 100, 1),
                    "mode": "GUIDED" if state.landed_state == airsim.LandedState.Flying else "LANDED",
                    "speed": round(airsim.Vector3r.get_length(state.kinematics_estimated.linear_velocity), 2)
                }
                
                topic = f"drone/{self.callsign}/telemetry"
                self.mqtt_client.publish(topic, json.dumps(payload))
                
                time.sleep(1.0) 
            except Exception as e:
                print(f"Error in telemetry loop: {e}")
                time.sleep(5) 
