"""
ai/detection/data_serve.py  — FastAPI ONNX detector that publishes to MQTT.

Notes:
 - The ONNX output format varies depending on export; this server inspects sess.get_outputs()
   and tries common YOLOv8/Ultralytics output shapes. If needed, adapt `parse_onnx_output`.
 - It publishes detections to MQTT topic: perception/{camera_id}/detections
"""

import os
import io
import time
import json
import logging
from typing import Tuple, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import onnxruntime as ort
import base64
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# Logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("detector")

# Config via env / .env
MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/resqron_yolov8n/weights/best.onnx")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", None)
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", None)
SCORE_THRESH = float(os.getenv("SCORE_THRESH", "0.25"))
CAMERA_ID_DEFAULT = os.getenv("CAMERA_ID", "cam-1")
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" or "gpu"
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
NMS_IOU = float(os.getenv("NMS_IOU", "0.45"))
WARMUP_ITERS = int(os.getenv("WARMUP_ITERS", "3"))
NAMES_PATH = os.getenv("NAMES_PATH", "./names.txt")  # optional mapping file

app = FastAPI(title="ResQron YOLO ONNX Detector")

# --- MQTT client setup ---------------------------------------
mqttc = mqtt.Client()
if MQTT_USERNAME:
    mqttc.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
try:
    mqttc.connect(MQTT_HOST, MQTT_PORT, 60)
    mqttc.loop_start()
    logger.info(f"MQTT connected to {MQTT_HOST}:{MQTT_PORT}")
except Exception as e:
    logger.error("Failed to connect to MQTT broker: %s", e)
    # keep running; publish will fail later with clear error

# --- load class names if present -----------------------------
CLASS_NAMES: List[str] = []
if os.path.exists(NAMES_PATH):
    try:
        CLASS_NAMES = [x.strip() for x in open(NAMES_PATH, "r", encoding="utf-8").read().splitlines() if x.strip()]
        logger.info("Loaded %d class names from %s", len(CLASS_NAMES), NAMES_PATH)
    except Exception as e:
        logger.warning("Failed to read names file %s: %s", NAMES_PATH, e)


# --- ONNX runtime session creation ---------------------------
def create_session(model_path: str, device: str = "cpu"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    providers = ["CPUExecutionProvider"]
    if device and device.lower().startswith("gpu"):
        # try CUDA provider first, fallback to CPU
        cuda = "CUDAExecutionProvider"
        providers = [cuda, "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
        logger.info("ONNX session created (providers=%s)", sess.get_providers())
        return sess
    except Exception as e:
        # Try CPU fallback if GPU failed
        if "CUDA" in str(e) and "CPUExecutionProvider" not in providers:
            logger.warning("CUDA provider failed, retrying with CPUExecutionProvider")
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            return sess
        logger.exception("Failed to create ONNX InferenceSession: %s", e)
        raise


# Create session (will raise if model not found)
try:
    sess = create_session(MODEL_PATH, DEVICE)
except Exception as e:
    logger.error("Cannot start detector without a valid ONNX model: %s", e)
    raise

# Print model input/output info for debugging
try:
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    logger.info("Model inputs: %s", [(i.name, i.shape, i.type) for i in inps])
    logger.info("Model outputs: %s", [(o.name, o.shape, o.type) for o in outs])
except Exception:
    logger.debug("Could not inspect model IO metadata.")

# Warmup to stabilize runtime kernels
def warmup(session, iters: int = 2, img_size: int = IMG_SIZE):
    try:
        dummy = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
        input_name = session.get_inputs()[0].name
        logger.info("Warming up ONNX session with %d iterations", iters)
        for _ in range(iters):
            session.run(None, {input_name: dummy})
    except Exception as e:
        logger.warning("Warmup failed: %s", e)

try:
    warmup(sess, WARMUP_ITERS, IMG_SIZE)
except Exception:
    pass


# --- image preprocessing / postprocessing --------------------
def preprocess_pil(img: Image.Image, imgsz: int = IMG_SIZE) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Resize and pad image to square (imgsz x imgsz). Paste is top-left (0,0).
    Returns tensor (1,3,H,W), original size (w0,h0), resized inner (nw,nh)
    """
    im = img.convert("RGB")
    w0, h0 = im.size
    r = imgsz / max(w0, h0)
    nw, nh = int(w0 * r), int(h0 * r)
    im_resized = im.resize((nw, nh), Image.BILINEAR)
    new_im = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    new_im.paste(im_resized, (0, 0))
    arr = np.array(new_im).astype(np.float32)
    arr = arr.transpose(2, 0, 1) / 255.0  # CHW, normalized
    arr = np.expand_dims(arr, 0)
    return arr, (w0, h0), (nw, nh)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45):
    # boxes: (N,4) x1,y1,x2,y2 ; scores: (N,)
    # returns indices to keep
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def parse_onnx_output(outputs: List[np.ndarray], orig_size: Tuple[int, int], resized_size: Tuple[int, int], score_thr: float = SCORE_THRESH) -> List[Dict[str, Any]]:
    """
    Attempt to parse common YOLO ONNX outputs into a list of detections with
    keys: xmin,ymin,xmax,ymax,conf,class
    """
    out0 = outputs[0]
    # unify shape to (N, C) or (N, 6)
    if isinstance(out0, np.ndarray) and out0.ndim == 3 and out0.shape[0] == 1:
        out = out0[0]
    elif isinstance(out0, np.ndarray) and out0.ndim == 2:
        out = out0
    else:
        out = out0.reshape(-1, out0.shape[-1])

    # Typical Ultralytics: rows = [x1,y1,x2,y2,score,class_conf,class_id?] or [x1,y1,x2,y2,confidence, class_id]
    detections = []
    w0, h0 = orig_size
    nw, nh = resized_size
    # choose strategy based on columns
    cols = out.shape[1]
    for row in out:
        if cols >= 6:
            # assume format x1,y1,x2,y2,score,class_id OR x1,y1,x2,y2,conf,class_conf,class_id
            x1, y1, x2, y2 = row[0:4].astype(float)
            # Some exports put center/wh — detect that (if x2 < x1 then center/wh)
            if x2 < x1:
                # treat as center,w,h
                cx, cy, w, h = x1, y1, x2, y2
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            # confidence selection
            conf = float(row[4])
            # class id detection
            if cols >= 7:
                class_id = int(round(row[6])) if row[6] is not None else 0
            else:
                class_id = int(round(row[5]))
        else:
            continue

        if conf < score_thr:
            continue

        # scale back to original image coordinates (we pasted at 0,0)
        scale_x = w0 / float(nw) if nw != 0 else 1.0
        scale_y = h0 / float(nh) if nh != 0 else 1.0
        gx1 = max(0, min(w0, x1 * scale_x))
        gy1 = max(0, min(h0, y1 * scale_y))
        gx2 = max(0, min(w0, x2 * scale_x))
        gy2 = max(0, min(h0, y2 * scale_y))

        detections.append({
            "xmin": float(gx1),
            "ymin": float(gy1),
            "xmax": float(gx2),
            "ymax": float(gy2),
            "conf": float(conf),
            "class": int(class_id)
        })

    # Optional NMS (if ONNX export didn't apply it)
    if len(detections) == 0:
        return []
    boxes_np = np.array([[d["xmin"], d["ymin"], d["xmax"], d["ymax"]] for d in detections])
    scores_np = np.array([d["conf"] for d in detections])
    keep_idx = non_max_suppression(boxes_np, scores_np, iou_threshold=NMS_IOU)
    final = [detections[i] for i in keep_idx]
    return final


# --- API endpoints --------------------------------------------
@app.post("/detect")
async def detect_file(file: UploadFile = File(...), camera_id: str = Form(CAMERA_ID_DEFAULT)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    arr, orig_size, resized_size = preprocess_pil(img, imgsz=IMG_SIZE)
    input_name = sess.get_inputs()[0].name
    t0 = time.time()
    try:
        outs = sess.run(None, {input_name: arr})
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    t1 = time.time()

    detections = parse_onnx_output(outs, orig_size, resized_size)
    # attach human-readable labels if available
    for d in detections:
        d["label"] = CLASS_NAMES[d["class"]] if d["class"] < len(CLASS_NAMES) else str(d["class"])

    payload = {
        "camera_id": camera_id,
        "timestamp": int(time.time() * 1000),
        "detections": detections,
        "processing_ms": int((t1 - t0) * 1000),
        "model": os.path.basename(MODEL_PATH)
    }

    topic = f"perception/{camera_id}/detections"
    try:
        mqttc.publish(topic, json.dumps(payload))
    except Exception as e:
        logger.warning("Failed to publish MQTT message: %s", e)

    return payload


@app.post("/detect_base64")
async def detect_b64(image_base64: str = Form(...), camera_id: str = Form(CAMERA_ID_DEFAULT)):
    try:
        imgdata = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(imgdata)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    arr, orig_size, resized_size = preprocess_pil(img, imgsz=IMG_SIZE)
    input_name = sess.get_inputs()[0].name
    t0 = time.time()
    try:
        outs = sess.run(None, {input_name: arr})
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    t1 = time.time()

    detections = parse_onnx_output(outs, orig_size, resized_size)
    for d in detections:
        d["label"] = CLASS_NAMES[d["class"]] if d["class"] < len(CLASS_NAMES) else str(d["class"])

    payload = {
        "camera_id": camera_id,
        "timestamp": int(time.time() * 1000),
        "detections": detections,
        "processing_ms": int((t1 - t0) * 1000),
        "model": os.path.basename(MODEL_PATH)
    }
    topic = f"perception/{camera_id}/detections"
    try:
        mqttc.publish(topic, json.dumps(payload))
    except Exception as e:
        logger.warning("Failed to publish MQTT message: %s", e)
    return payload


@app.get("/health")
async def health():
    ok = os.path.exists(MODEL_PATH)
    return {
        "ok": ok,
        "model": MODEL_PATH,
        "device": DEVICE,
        "class_count": len(CLASS_NAMES),
        "onnx_providers": sess.get_providers()
    }
