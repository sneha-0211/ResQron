from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn, random, base64, io, os
from PIL import Image
import numpy as np

_has_transformers = False
_has_onnx = False
try:
    from transformers import pipeline
    _has_transformers = True
except Exception:
    pass
try:
    import onnxruntime as ort  # noqa: F401
    _has_onnx = True
except Exception:
    pass

app = FastAPI()

class DisasterDetection(BaseModel):
    label: str
    confidence: float
    severity: str
    coordinates: list
    description: str
    recommended_actions: list

class ImageAnalysisRequest(BaseModel):
    imageUrl: str = None
    imageData: str = None  # base64 encoded image
    coordinates: list = None

class TextClassifyRequest(BaseModel):
    text: str
    top_k: int = 3

class TextClassifyResponse(BaseModel):
    labels: list
    scores: list


def _simple_text_rules(text: str, top_k: int = 3):
    text_l = text.lower()
    rules = {
        'earthquake': ['quake', 'richter', 'aftershock', 'tremor', 'seismic'],
        'flood': ['flood', 'inundat', 'overflow', 'water level', 'levee'],
        'fire': ['fire', 'wildfire', 'smoke', 'blaze', 'burn'],
        'landslide': ['landslide', 'mudslide', 'debris flow'],
        'cyclone': ['cyclone', 'hurricane', 'typhoon', 'storm surge']
    }
    scores = []
    for label, kws in rules.items():
        score = 0.0
        for kw in kws:
            if kw in text_l:
                score += 0.2
        if score > 0:
            scores.append((label, min(score, 0.99)))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        scores = [('other', 0.5)]
    labels = [s[0] for s in scores[:top_k]]
    vals = [float(s[1]) for s in scores[:top_k]]
    return TextClassifyResponse(labels=labels, scores=vals)


@app.post("/classify-text", response_model=TextClassifyResponse)
async def classify_text(req: TextClassifyRequest):
    try:
        if _has_transformers:
            clf = pipeline('text-classification', model=os.getenv('BERT_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english'), top_k=req.top_k)
            out = clf(req.text)
            # transformers returns list of lists when top_k>1
            if isinstance(out, list) and out and isinstance(out[0], list):
                labels = [x['label'] for x in out[0]]
                scores = [float(x['score']) for x in out[0]]
            else:
                labels = [out[0]['label']]
                scores = [float(out[0]['score'])]
            return TextClassifyResponse(labels=labels, scores=scores)
        # fallback rules
        return _simple_text_rules(req.text, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text classify failed: {e}")


class DetectObjectsResponse(BaseModel):
    boxes: list  # list of {x1,y1,x2,y2, label, score}


@app.post("/detect-objects", response_model=DetectObjectsResponse)
async def detect_objects(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        # Try ONNX runtime path if available, else stub
        if _has_onnx:
            pass
        # Simple stub: return no boxes
        return DetectObjectsResponse(boxes=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detect failed: {e}")

# Enhanced disaster detection with more realistic scenarios
DISASTER_TYPES = {
    "flood": {
        "severity_levels": ["low", "moderate", "high", "critical"],
        "descriptions": {
            "low": "Minor flooding detected - water level rise",
            "moderate": "Moderate flooding - significant water accumulation",
            "high": "Severe flooding - widespread water damage",
            "critical": "Critical flooding - life-threatening conditions"
        },
        "actions": ["Evacuate area", "Deploy rescue teams", "Set up emergency shelters", "Monitor water levels"]
    },
    "fire": {
        "severity_levels": ["smoke", "small_fire", "large_fire", "wildfire"],
        "descriptions": {
            "smoke": "Smoke detected - potential fire hazard",
            "small_fire": "Small fire detected - localized burning",
            "large_fire": "Large fire detected - significant damage",
            "wildfire": "Wildfire detected - spreading rapidly"
        },
        "actions": ["Alert fire department", "Evacuate nearby areas", "Deploy fire suppression", "Monitor wind conditions"]
    },
    "earthquake": {
        "severity_levels": ["minor", "moderate", "strong", "severe"],
        "descriptions": {
            "minor": "Minor ground movement detected",
            "moderate": "Moderate earthquake - structural damage possible",
            "strong": "Strong earthquake - significant damage likely",
            "severe": "Severe earthquake - catastrophic damage"
        },
        "actions": ["Check structural integrity", "Evacuate buildings", "Deploy search and rescue", "Assess infrastructure damage"]
    },
    "landslide": {
        "severity_levels": ["minor", "moderate", "major", "catastrophic"],
        "descriptions": {
            "minor": "Minor slope movement detected",
            "moderate": "Moderate landslide - localized damage",
            "major": "Major landslide - significant terrain change",
            "catastrophic": "Catastrophic landslide - massive destruction"
        },
        "actions": ["Evacuate affected areas", "Deploy rescue teams", "Assess stability", "Monitor for further movement"]
    }
}

def analyze_disaster_severity(disaster_type, confidence):
    """Determine severity based on confidence and disaster type"""
    if confidence < 0.7:
        return "low" if disaster_type != "fire" else "smoke"
    elif confidence < 0.85:
        return "moderate" if disaster_type != "fire" else "small_fire"
    elif confidence < 0.95:
        return "high" if disaster_type != "fire" else "large_fire"
    else:
        return "critical" if disaster_type != "fire" else "wildfire"

@app.post("/analyze", response_model=DisasterDetection)
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read image data
        image_data = await file.read()
        
        # In production, you would use your actual ML model here
        # For demo, we'll simulate more realistic detection
        
        # Simulate different disaster types with weighted probabilities
        disaster_weights = {
            "flood": 0.3,
            "fire": 0.25, 
            "earthquake": 0.2,
            "landslide": 0.15,
            "other": 0.1
        }
        
        # Weighted random selection
        rand = random.random()
        cumulative = 0
        label = "other"
        for disaster, weight in disaster_weights.items():
            cumulative += weight
            if rand <= cumulative:
                label = disaster
                break
        
        confidence = round(random.uniform(0.6, 0.98), 2)
        severity = analyze_disaster_severity(label, confidence)
        
        # Get disaster info
        disaster_info = DISASTER_TYPES.get(label, {
            "severity_levels": ["unknown"],
            "descriptions": {"unknown": "Unknown disaster type detected"},
            "actions": ["Investigate further", "Deploy assessment team"]
        })
        
        description = disaster_info["descriptions"].get(severity, "Disaster detected")
        recommended_actions = disaster_info["actions"]
        
        # Simulate coordinates (in production, this would come from GPS data)
        coordinates = [round(random.uniform(28.0, 29.0), 6), round(random.uniform(77.0, 78.0), 6)]
        
        return DisasterDetection(
            label=label,
            confidence=confidence,
            severity=severity,
            coordinates=coordinates,
            description=description,
            recommended_actions=recommended_actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/analyze-url", response_model=DisasterDetection)
async def analyze_image_url(request: ImageAnalysisRequest):
    """Analyze image from URL or base64 data"""
    try:
        # Simulate analysis (in production, download and process the image)
        disaster_weights = {
            "flood": 0.3,
            "fire": 0.25, 
            "earthquake": 0.2,
            "landslide": 0.15,
            "other": 0.1
        }
        
        rand = random.random()
        cumulative = 0
        label = "other"
        for disaster, weight in disaster_weights.items():
            cumulative += weight
            if rand <= cumulative:
                label = disaster
                break
        
        confidence = round(random.uniform(0.6, 0.98), 2)
        severity = analyze_disaster_severity(label, confidence)
        
        disaster_info = DISASTER_TYPES.get(label, {
            "severity_levels": ["unknown"],
            "descriptions": {"unknown": "Unknown disaster type detected"},
            "actions": ["Investigate further", "Deploy assessment team"]
        })
        
        description = disaster_info["descriptions"].get(severity, "Disaster detected")
        recommended_actions = disaster_info["actions"]
        
        # Use provided coordinates or generate random ones
        coordinates = request.coordinates or [round(random.uniform(28.0, 29.0), 6), round(random.uniform(77.0, 78.0), 6)]
        
        return DisasterDetection(
            label=label,
            confidence=confidence,
            severity=severity,
            coordinates=coordinates,
            description=description,
            recommended_actions=recommended_actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "disaster-detection-ai"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
