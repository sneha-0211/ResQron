"""
This script sets up and configures all the AI components for real-time drone integration:
- MiDaS depth estimation service
- A* path planning service  
- Backend MQTT integration
- Dashboard real-time updates
"""

import os
import sys
import time
import json
import subprocess
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional

class ResQronIntegration:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.services = {
            'depth_service': {
                'name': 'MiDaS Depth Estimation',
                'port': 8001,
                'script': 'ai/depth/midas_infer.py',
                'args': ['service'],
                'health_endpoint': '/health'
            },
            'planner_service': {
                'name': 'A* Path Planner',
                'port': 8000,
                'script': 'ai/planner/api.py',
                'args': [],
                'health_endpoint': '/system_status'
            },
            'backend_service': {
                'name': 'Backend Server',
                'port': 5000,
                'script': 'backend/server.js',
                'args': [],
                'health_endpoint': '/api/system/health'
            },
            'dashboard_service': {
                'name': 'Dashboard',
                'port': 3000,
                'script': 'dashboard',
                'args': ['npm', 'run', 'dev'],
                'health_endpoint': '/'
            }
        }
        self.processes = {}

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        print("Checking dependencies...")
        
        # Check Python dependencies
        python_deps = [
            'torch', 'torchvision', 'opencv-python', 'fastapi', 'uvicorn',
            'paho-mqtt', 'shapely', 'timm', 'numpy', 'pillow'
        ]
        
        missing_python = []
        for dep in python_deps:
            try:
                __import__(dep.replace('-', '_'))
            except ImportError:
                missing_python.append(dep)
        
        if missing_python:
            print(f"Missing Python dependencies: {', '.join(missing_python)}")
            print("Run: pip install -r ai/requirements.txt")
            return False

        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Node.js not found")
                return False
        except FileNotFoundError:
            print("Node.js not found")
            return False
        
        # Check if MQTT broker is available
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client()
            client.connect('localhost', 1883, 5)
            client.disconnect()
        except:
            print("MQTT broker not available on localhost:1883")
            print("Start MQTT broker: docker run -it -p 1883:1883 eclipse-mosquitto")
        
        print("Dependencies check completed")
        return True

    def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            print(f"Unknown service: {service_name}")
            return False
        
        service = self.services[service_name]
        print(f"Starting {service['name']}...")
        
        try:
            if service_name == 'dashboard_service':
                process = subprocess.Popen(
                    service['args'],
                    cwd=self.base_dir / 'dashboard',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                script_path = self.base_dir / service['script']
                if service_name == 'backend_service':
                    process = subprocess.Popen(
                        ['node', str(script_path)] + service['args'],
                        cwd=self.base_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, str(script_path)] + service['args'],
                        cwd=self.base_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
            
            self.processes[service_name] = process
            print(f"{service['name']} started (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"Failed to start {service['name']}: {e}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.processes:
            print(f"Service {service_name} not running")
            return True
        
        process = self.processes[service_name]
        print(f"Stopping {self.services[service_name]['name']}...")
        
        try:
            process.terminate()
            process.wait(timeout=10)
            del self.processes[service_name]
            print(f"{self.services[service_name]['name']} stopped")
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            del self.processes[service_name]
            print(f"{self.services[service_name]['name']} force stopped")
            return True
        except Exception as e:
            print(f"Failed to stop {service_name}: {e}")
            return False

    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        try:
            response = requests.get(
                f"http://localhost:{service['port']}{service['health_endpoint']}",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def start_all_services(self) -> bool:
        """Start all services in the correct order."""
        print("Starting all ResQron AI services...")
        
        # Start services in dependency order
        service_order = ['depth_service', 'planner_service', 'backend_service', 'dashboard_service']
        
        for service_name in service_order:
            if not self.start_service(service_name):
                print(f"Failed to start {service_name}, stopping all services")
                self.stop_all_services()
                return False
            
            # Wait for service to be ready
            print(f"Waiting for {self.services[service_name]['name']} to be ready...")
            for _ in range(30): 
                if self.check_service_health(service_name):
                    print(f"{self.services[service_name]['name']} is ready")
                    break
                time.sleep(1)
            else:
                print(f"{self.services[service_name]['name']} may not be ready yet")
        
        print("All services started successfully!")
        return True

    def stop_all_services(self) -> bool:
        """Stop all running services."""
        print("Stopping all services...")
        
        success = True
        for service_name in list(self.processes.keys()):
            if not self.stop_service(service_name):
                success = False
        
        return success

    def check_all_status(self) -> Dict[str, bool]:
        """Check status of all services."""
        print("Checking service status...")
        
        status = {}
        for service_name, service in self.services.items():
            is_running = service_name in self.processes
            is_healthy = self.check_service_health(service_name) if is_running else False
            
            status[service_name] = {
                'running': is_running,
                'healthy': is_healthy,
                'name': service['name'],
                'port': service['port']
            }
            
            status_icon = "OK" if is_healthy else "RUNNING" if is_running else "STOPPED"
            print(f"{status_icon} {service['name']} (Port {service['port']}): {'Healthy' if is_healthy else 'Running' if is_running else 'Stopped'}")
        
        return status

    def create_integration_test(self) -> str:
        """Create an integration test script."""
        test_script = """
\"\"\"
ResQron AI Integration Test

This script tests the integration between all AI components.
\"\"\"

import requests
import json
import time

def test_depth_service():
    \"\"\"Test MiDaS depth estimation service.\"\"\"
    try:
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            print("Depth service is healthy")
            return True
    except:
        pass
    print("Depth service is not responding")
    return False

def test_planner_service():
    \"\"\"Test A* path planning service.\"\"\"
    try:
        response = requests.get("http://localhost:8000/system_status")
        if response.status_code == 200:
            print("Planner service is healthy")
            return True
    except:
        pass
    print("Planner service is not responding")
    return False

def test_backend_service():
    \"\"\"Test backend server.\"\"\"
    try:
        response = requests.get("http://localhost:5000/api/system/health")
        if response.status_code == 200:
            print("Backend service is healthy")
            return True
    except:
        pass
    print("Backend service is not responding")
    return False

def test_dashboard():
    \"\"\"Test dashboard.\"\"\"
    try:
        response = requests.get("http://localhost:3000/")
        if response.status_code == 200:
            print("Dashboard is accessible")
            return True
    except:
        pass
    print("Dashboard is not accessible")
    return False

def main():
    print("Running ResQron AI Integration Tests...")
    
    tests = [
        test_depth_service,
        test_planner_service,
        test_backend_service,
        test_dashboard
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\\nTest Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All integration tests passed!")
        return True
    else:
        print("Some integration tests failed")
        return False

if __name__ == "__main__":
    main()
"""
        
        test_file = self.base_dir / "integration_test.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        # Make it executable
        os.chmod(test_file, 0o755)
        return str(test_file)

    def create_docker_compose(self) -> str:
        """Create a Docker Compose file for easy deployment."""
        docker_compose = """
version: '3.8'

services:
  mosquitto:
    image: eclipse-mosquitto:2.0
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./ops/mosquitto/config:/mosquitto/config
    command: mosquitto -c /mosquitto/config/mosquitto.conf

  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password

  depth-service:
    build:
      context: .
      dockerfile: ai/Dockerfile
    ports:
      - "8001:8001"
    environment:
      - MQTT_HOST=mosquitto
      - MQTT_PORT=1883
    depends_on:
      - mosquitto
    command: python ai/depth/midas_infer.py service

  planner-service:
    build:
      context: .
      dockerfile: ai/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MQTT_HOST=mosquitto
      - MQTT_PORT=1883
    depends_on:
      - mosquitto
    command: python ai/planner/api.py

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "5000:5000"
    environment:
      - MQTT_URL=mqtt://mosquitto:1883
      - MONGO_URI=mongodb://admin:password@mongodb:27017/sih?authSource=admin
      - PLANNER_URL=http://planner-service:8000
    depends_on:
      - mosquitto
      - mongodb
      - planner-service

  dashboard:
    build:
      context: .
      dockerfile: dashboard/Dockerfile
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:5000
    depends_on:
      - backend

volumes:
  mongodb_data:
"""
        
        compose_file = self.base_dir / "docker-compose.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(docker_compose)
        
        return str(compose_file)

    def create_startup_script(self) -> str:
        """Create a startup script for easy service management."""
        startup_script = """#!/bin/bash
# ResQron AI Services Startup Script

set -e

echo "Starting ResQron AI Services..."

# Check if MQTT broker is running
if ! nc -z localhost 1883; then
    echo "MQTT broker not running. Starting with Docker..."
    docker run -d --name resqron-mqtt -p 1883:1883 eclipse-mosquitto:2.0
    sleep 5
fi

# Check if MongoDB is running
if ! nc -z localhost 27017; then
    echo "MongoDB not running. Starting with Docker..."
    docker run -d --name resqron-mongodb -p 27017:27017 mongo:7.0
    sleep 5
fi

# Start AI services
echo "Starting AI services..."
python integration_setup.py --start-all

echo "All services started!"
echo "Dashboard: http://localhost:3000"
echo "Backend API: http://localhost:5000"
echo "Planner API: http://localhost:8000"
echo "Depth API: http://localhost:8001"
"""
        
        script_file = self.base_dir / "start_services.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # Make it executable
        os.chmod(script_file, 0o755)
        return str(script_file)

def main():
    parser = argparse.ArgumentParser(description='ResQron AI Integration Setup')
    parser.add_argument('--start-all', action='store_true', help='Start all services')
    parser.add_argument('--stop-all', action='store_true', help='Stop all services')
    parser.add_argument('--check-status', action='store_true', help='Check status of all services')
    parser.add_argument('--create-files', action='store_true', help='Create integration files')
    parser.add_argument('--test', action='store_true', help='Run integration tests')
    
    args = parser.parse_args()
    
    integration = ResQronIntegration()
    
    if args.create_files:
        print("Creating integration files...")
        test_file = integration.create_integration_test()
        compose_file = integration.create_docker_compose()
        startup_script = integration.create_startup_script()
        print(f"Created files:")
        print(f"  - {test_file}")
        print(f"  - {compose_file}")
        print(f"  - {startup_script}")
    
    if args.check_status:
        integration.check_all_status()
    
    if args.start_all:
        if not integration.check_dependencies():
            print("Dependency check failed")
            sys.exit(1)
        integration.start_all_services()
    
    if args.stop_all:
        integration.stop_all_services()
    
    if args.test:
        test_file = integration.create_integration_test()
        subprocess.run([sys.executable, test_file])
    
    if not any([args.start_all, args.stop_all, args.check_status, args.create_files, args.test]):
        parser.print_help()

if __name__ == "__main__":
    main()
