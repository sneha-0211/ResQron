import cv2
import torch
import numpy as np
import json
import time
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", None)
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", None)
DEPTH_SERVICE_PORT = int(os.getenv("DEPTH_SERVICE_PORT", "8001"))

# --- Global Model Instance ---
midas_model = None
midas_transform = None

# --- MQTT Client Setup ---
mqtt_client = mqtt.Client()
if MQTT_USERNAME:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("MQTT connected successfully")
        client.subscribe("perception/+/detections")
        client.subscribe("depth/+/request")
    else:
        logger.error(f"MQTT connection failed with code {rc}")

def on_mqtt_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split('/')
        if len(topic_parts) >= 3 and topic_parts[0] == "perception" and topic_parts[2] == "detections":
            # Handle YOLO detections for depth estimation
            camera_id = topic_parts[1]
            payload = json.loads(msg.payload.decode())
            asyncio.create_task(process_detections_for_depth(camera_id, payload))
        elif len(topic_parts) >= 3 and topic_parts[0] == "depth" and topic_parts[2] == "request":
            # Handle direct depth estimation requests
            camera_id = topic_parts[1]
            payload = json.loads(msg.payload.decode())
            asyncio.create_task(handle_depth_request(camera_id, payload))
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

# --- FastAPI App ---
app = FastAPI(
    title="MiDaS Depth Estimation Service",
    description="Real-time depth estimation for drone perception pipeline",
    version="1.0.0"
)

# --- Data Models ---
class Detection(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    conf: float
    class_id: int
    label: str

class DepthRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    detections: List[Detection] = []
    camera_id: str = "default"

class DepthResponse(BaseModel):
    camera_id: str
    timestamp: int
    depth_map_path: Optional[str] = None
    depth_estimates: List[Dict[str, Any]] = []
    processing_ms: int

# --- Model Management ---

def load_model(model_type: str = "DPT_Large") -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Loads a pretrained MiDaS model from PyTorch Hub.

    Args:
        model_type (str): The specific MiDaS model to load. 
                          "DPT_Large" is a powerful and common choice.

    Returns:
        A tuple containing the loaded model and the required transforms for input images.
    """
    print(f"Loading MiDaS model: {model_type}...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type, force_reload=True)
        
        # Load the appropriate transforms for the chosen model
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.dpt_transform if "dpt" in model_type.lower() else transforms.midas_transform

        print("Model loaded successfully.")
        return model, transform
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and PyTorch is installed correctly.")
        raise

# --- Inference ---

def run_inference(model: torch.nn.Module, transform: torch.nn.Module, image_path: str) -> np.ndarray:
    """
    Runs depth estimation on a single image.

    Args:
        model: The loaded MiDaS model.
        transform: The corresponding transformation function for the model.
        image_path (str): The path to the input image file.

    Returns:
        A 2D NumPy array representing the depth map, where higher values are further away.
        Returns None if the image cannot be loaded.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image at {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Determine the device (use GPU if available)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # Transform the input image and move it to the selected device
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        
        # Resize the prediction to the original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Move the depth map back to the CPU and convert it to a NumPy array
    depth_map = prediction.cpu().numpy()
    
    return depth_map

# --- Bridging Gap: YOLO Integration ---

def get_depth_for_detections(depth_map: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> List[float]:
    """
    Calculates the average depth for a list of bounding boxes from a YOLO detector.

    Args:
        depth_map (np.ndarray): The depth map generated by run_inference.
        detections (List[Tuple[int, int, int, int]]): A list of bounding boxes,
            where each box is in the format (xmin, ymin, xmax, ymax).

    Returns:
        A list of floats, where each float is the average depth of the corresponding
        detection. Returns an empty list if no detections are provided.
    """
    if detections is None or not detections:
        return []

    depth_estimates = []
    for (xmin, ymin, xmax, ymax) in detections:
        # Ensure coordinates are within the bounds of the depth map
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(depth_map.shape[1], xmax), min(depth_map.shape[0], ymax)

        # Extract the region of interest (ROI) from the depth map
        depth_roi = depth_map[ymin:ymax, xmin:xmax]
        
        # Calculate the average depth, avoiding division by zero for empty ROIs
        if depth_roi.size > 0:
            average_depth = float(np.mean(depth_roi))
        else:
            average_depth = 0.0
            
        depth_estimates.append(average_depth)
        
    return depth_estimates

# --- Real-time Processing Functions ---

async def process_detections_for_depth(camera_id: str, detection_payload: Dict[str, Any]):
    """
    Process YOLO detections and generate depth estimates for each detection.
    Publishes results to MQTT topic: depth/{camera_id}/estimates
    """
    try:
        if not midas_model or not midas_transform:
            logger.error("MiDaS model not loaded")
            return

        # Extract image data from detection payload
        image_data = detection_payload.get('image_data')
        if not image_data:
            logger.warning(f"No image data in detection payload for camera {camera_id}")
            return

        # Decode base64 image
        import base64
        from PIL import Image
        import io
        
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Run depth inference
        start_time = time.time()
        depth_map = run_inference(midas_model, midas_transform, img_cv)
        processing_time = int((time.time() - start_time) * 1000)
        
        if depth_map is None:
            logger.error(f"Failed to generate depth map for camera {camera_id}")
            return

        # Process detections for depth estimates
        detections = detection_payload.get('detections', [])
        depth_estimates = []
        
        for detection in detections:
            bbox = (int(detection['xmin']), int(detection['ymin']), 
                   int(detection['xmax']), int(detection['ymax']))
            avg_depth = get_depth_for_detections(depth_map, [bbox])[0]
            
            depth_estimates.append({
                'detection_id': detection.get('id', 'unknown'),
                'bbox': bbox,
                'class': detection.get('class', 'unknown'),
                'confidence': detection.get('conf', 0.0),
                'average_depth': float(avg_depth),
                'depth_confidence': 'high' if avg_depth > 0 else 'low'
            })

        # Save depth map visualization
        depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_color = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)
        
        # Save to file
        timestamp = int(time.time() * 1000)
        depth_map_path = f"depth_maps/depth_{camera_id}_{timestamp}.png"
        os.makedirs("depth_maps", exist_ok=True)
        cv2.imwrite(depth_map_path, depth_map_color)

        # Publish depth estimates to MQTT
        depth_payload = {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'depth_map_path': depth_map_path,
            'depth_estimates': depth_estimates,
            'processing_ms': processing_time,
            'detection_count': len(detections)
        }

        topic = f"depth/{camera_id}/estimates"
        mqtt_client.publish(topic, json.dumps(depth_payload))
        logger.info(f"Published depth estimates for {len(detections)} detections from camera {camera_id}")

    except Exception as e:
        logger.error(f"Error processing detections for depth: {e}")

async def handle_depth_request(camera_id: str, request_payload: Dict[str, Any]):
    """
    Handle direct depth estimation requests via MQTT.
    """
    try:
        if not midas_model or not midas_transform:
            logger.error("MiDaS model not loaded")
            return

        # Process the depth request
        image_path = request_payload.get('image_path')
        image_base64 = request_payload.get('image_base64')
        
        if image_path and os.path.exists(image_path):
            depth_map = run_inference(midas_model, midas_transform, image_path)
        elif image_base64:
            import base64
            from PIL import Image
            import io
            
            img_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            depth_map = run_inference(midas_model, midas_transform, img_cv)
        else:
            logger.error("No valid image data in depth request")
            return

        if depth_map is None:
            logger.error("Failed to generate depth map")
            return

        # Process any detections in the request
        detections = request_payload.get('detections', [])
        depth_estimates = []
        
        for detection in detections:
            bbox = (int(detection['xmin']), int(detection['ymin']), 
                   int(detection['xmax']), int(detection['ymax']))
            avg_depth = get_depth_for_detections(depth_map, [bbox])[0]
            
            depth_estimates.append({
                'bbox': bbox,
                'class': detection.get('class', 'unknown'),
                'confidence': detection.get('conf', 0.0),
                'average_depth': float(avg_depth)
            })

        # Save depth map
        timestamp = int(time.time() * 1000)
        depth_map_path = f"depth_maps/request_{camera_id}_{timestamp}.png"
        os.makedirs("depth_maps", exist_ok=True)
        
        depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_color = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)
        cv2.imwrite(depth_map_path, depth_map_color)

        # Publish response
        response_payload = {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'depth_map_path': depth_map_path,
            'depth_estimates': depth_estimates,
            'processing_ms': 0,  # Could be calculated if needed
            'status': 'success'
        }

        topic = f"depth/{camera_id}/response"
        mqtt_client.publish(topic, json.dumps(response_payload))
        logger.info(f"Published depth response for camera {camera_id}")

    except Exception as e:
        logger.error(f"Error handling depth request: {e}")

# --- FastAPI Endpoints ---

@app.post("/estimate_depth", response_model=DepthResponse)
async def estimate_depth_endpoint(request: DepthRequest):
    """
    FastAPI endpoint for depth estimation with YOLO detections.
    """
    try:
        if not midas_model or not midas_transform:
            raise HTTPException(status_code=500, detail="MiDaS model not loaded")

        start_time = time.time()
        
        # Process image
        if request.image_path and os.path.exists(request.image_path):
            depth_map = run_inference(midas_model, midas_transform, request.image_path)
        elif request.image_base64:
            import base64
            from PIL import Image
            import io
            
            img_bytes = base64.b64decode(request.image_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            depth_map = run_inference(midas_model, midas_transform, img_cv)
        else:
            raise HTTPException(status_code=400, detail="No valid image data provided")

        if depth_map is None:
            raise HTTPException(status_code=500, detail="Failed to generate depth map")

        # Process detections
        depth_estimates = []
        for detection in request.detections:
            bbox = (int(detection.xmin), int(detection.ymin), 
                   int(detection.xmax), int(detection.ymax))
            avg_depth = get_depth_for_detections(depth_map, [bbox])[0]
            
            depth_estimates.append({
                'bbox': bbox,
                'class': detection.label,
                'confidence': detection.conf,
                'average_depth': float(avg_depth)
            })

        # Save depth map
        timestamp = int(time.time() * 1000)
        depth_map_path = f"depth_maps/api_{request.camera_id}_{timestamp}.png"
        os.makedirs("depth_maps", exist_ok=True)
        
        depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_color = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)
        cv2.imwrite(depth_map_path, depth_map_color)

        processing_time = int((time.time() - start_time) * 1000)

        return DepthResponse(
            camera_id=request.camera_id,
            timestamp=timestamp,
            depth_map_path=depth_map_path,
            depth_estimates=depth_estimates,
            processing_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error in depth estimation endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if midas_model is not None else "unhealthy",
        "model_loaded": midas_model is not None,
        "mqtt_connected": mqtt_client.is_connected(),
        "timestamp": int(time.time() * 1000)
    }

# --- Service Initialization ---

def initialize_service():
    """Initialize the depth estimation service."""
    global midas_model, midas_transform
    
    try:
        # Load MiDaS model
        logger.info("Loading MiDaS model...")
        midas_model, midas_transform = load_model()
        logger.info("MiDaS model loaded successfully")
        
        # Try to connect to MQTT, but don't fail if it's not available
        logger.info(f"Attempting to connect to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
            mqtt_client.loop_start()
            logger.info("MQTT connected successfully")
        except Exception as mqtt_error:
            logger.warning(f"MQTT connection failed: {mqtt_error}")
            logger.warning("Continuing without MQTT - some features will be limited")
        
        # Create output directory
        os.makedirs("depth_maps", exist_ok=True)
        
        logger.info("Depth estimation service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

# --- Example Usage ---

def main():
    """
    Main function to demonstrate the depth estimation pipeline.
    This example shows how to load the model, run inference on a test image,
    and get depth estimates for predefined bounding boxes.
    """
    # Create a dummy image for testing if it doesn't exist
    test_image_path = "test_image.jpg"
    if not Path(test_image_path).exists():
        print("Creating a dummy test image...")
        dummy_img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, dummy_img)

    # 1. Load the MiDaS model
    try:
        midas_model, midas_transform = load_model()
    except Exception:
        print("Could not run main example due to model loading failure.")
        return

    # 2. Run depth inference on the image
    print(f"\nRunning inference on '{test_image_path}'...")
    depth_map = run_inference(midas_model, midas_transform, test_image_path)
    
    if depth_map is not None:
        # Normalize depth map for visualization (0-255)
        depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_color = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)
        
        output_path = "depth_map_output.png"
        cv2.imwrite(output_path, depth_map_color)
        print(f"Depth map saved to '{output_path}'")

        # 3. Get depth for sample YOLO detections
        # These would come from your YOLO model in a real application
        sample_detections = [
            (100, 150, 200, 350),  # A person-like bounding box
            (300, 250, 450, 400),  # A vehicle-like bounding box
        ]
        
        print(f"\nGetting depth for {len(sample_detections)} sample detections...")
        estimated_depths = get_depth_for_detections(depth_map, sample_detections)
        
        for i, depth in enumerate(estimated_depths):
            # Note: The depth value is relative; it's not in meters.
            # It requires calibration or context to be converted to an absolute distance.
            print(f"  - Detection {i+1}: Average depth value = {depth:.2f}")

def run_service():
    """Run the depth estimation service as a FastAPI application."""
    import uvicorn
    
    # Initialize the service
    initialize_service()
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=DEPTH_SERVICE_PORT,
        log_level="info"
    )

if __name__ == '__main__':
    from pathlib import Path
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "service":
        # Run as a service
        run_service()
    else:
        # Run the example
        main()
