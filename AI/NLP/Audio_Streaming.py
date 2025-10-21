from __future__ import annotations
import os
import sys
import time
import uuid
import json
import queue
import math
import glob
import logging
import signal
import threading
import argparse
import tempfile
import subprocess
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

# third-party
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import pyttsx3
import paho.mqtt.client as mqtt

# faster-whisper ASR
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# Resemblyzer for speaker embeddings
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except Exception:
    RESEMBLYZER_AVAILABLE = False

# Sentence transformers for lightweight NLU embeddings
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# denoising
try:
    import noisereduce as nr
    import resampy
    NOISEREDUCE_AVAILABLE = True
except Exception:
    NOISEREDUCE_AVAILABLE = False

# MAVLink optional
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except Exception:
    MAVLINK_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RescueDroneChatbot")

# ---------------- CONFIG (env + CLI defaults) ----------------
DEFAULT_SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
DEFAULT_CHANNELS = int(os.environ.get("CHANNELS", "1"))
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "en")

DEFAULT_VOSK_MODELS = {}  # kept for backwards compat if desired

SPK_MODEL_PATH = os.environ.get("SPK_MODEL_PATH", "./models/vosk-spk")
KNOWN_SPEAKERS_DIR = os.environ.get("KNOWN_SPEAKERS_DIR", "./known_speakers")
AUDIO_SAVE_DIR = os.environ.get("AUDIO_SAVE_DIR", "./captured_audio")

# MQTT defaults
DEFAULT_MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
DEFAULT_MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC_ALERT = os.environ.get("MQTT_TOPIC_ALERT", "sih/rescue/alerts")
MQTT_TOPIC_CMD = os.environ.get("MQTT_TOPIC_CMD", "sih/rescue/drone_cmd")

# Wakeword & VAD
WAKEWORD = os.environ.get("WAKEWORD", "rescue")
USE_WAKEWORD = os.environ.get("USE_WAKEWORD", "1") == "1"
USE_VAD = os.environ.get("USE_VAD", "1") == "1"
VAD_MODE = int(os.environ.get("VAD_MODE", "2"))

AUDIO_PRE_SECONDS = float(os.environ.get("AUDIO_PRE_SECONDS", "2"))
AUDIO_POST_SECONDS = float(os.environ.get("AUDIO_POST_SECONDS", "2"))

ESCALATION_WINDOW = int(os.environ.get("ESCALATION_WINDOW", "120"))
ESCALATION_GEOFENCE_RADIUS_M = int(os.environ.get("ESCALATION_GEOFENCE_RADIUS_M", "200"))

SPEAKER_SIM_THRESHOLD = float(os.environ.get("SPEAKER_SIM_THRESHOLD", "0.75"))
MAX_LANG_HISTORY = int(os.environ.get("MAX_LANG_HISTORY", "5"))

RESPONSE_POLICY = os.environ.get("RESPONSE_POLICY", "en_only")

# ASR worker defaults
ASR_MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "small")  # tiny, small, medium
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8");

# ---------------- GLOBALS ----------------
running = True
asr_input_q = None  # multiprocessing queue (set in main)
asr_output_q = None

tts_queue: queue.Queue = queue.Queue()
offline_mqtt_queue: queue.Queue = queue.Queue()

audio_ring = deque(maxlen=int(DEFAULT_SAMPLE_RATE * AUDIO_PRE_SECONDS))

# ASR model handle will be in child process
encoder = None  # resemblyzer encoder (in main process)
embed_model = None
intent_clf = None

mqtt_client = None

# ---------------- UTILITIES ----------------

def ensure_dirs():
    os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
    os.makedirs(KNOWN_SPEAKERS_DIR, exist_ok=True)

# denoise helper (best-effort)
def denoise_audio_file(in_path: str, out_path: str, target_sr:int=16000):
    if not NOISEREDUCE_AVAILABLE:
        # fallback: copy
        import shutil
        shutil.copy(in_path, out_path)
        return out_path
    audio, sr = sf.read(in_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    reduced = nr.reduce_noise(y=audio, sr=sr)
    if sr != target_sr:
        reduced = resampy.resample(reduced, sr, target_sr)
    sf.write(out_path, reduced, target_sr)
    return out_path

def haversine_meters(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371000
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(x))

# ---------------- MQTT with TLS support ----------------

def mqtt_setup(broker: str, port: int, username: Optional[str]=None, password: Optional[str]=None, tls_ca: Optional[str]=None, tls_cert: Optional[str]=None, tls_key: Optional[str]=None):
    global mqtt_client
    mqtt_client = mqtt.Client(client_id=f"device-{uuid.uuid4().hex[:8]}", protocol=mqtt.MQTTv311)
    if username and password:
        mqtt_client.username_pw_set(username, password)
    if tls_ca or tls_cert or tls_key:
        # Only set TLS parameters when provided
        try:
            if tls_cert and tls_key:
                mqtt_client.tls_set(ca_certs=tls_ca if tls_ca else None, certfile=tls_cert, keyfile=tls_key)
            else:
                mqtt_client.tls_set(ca_certs=tls_ca)
            mqtt_client.tls_insecure_set(False)
        except Exception:
            logger.exception("Failed to configure TLS for MQTT")
    mqtt_client.on_connect = lambda c, u, f, rc: logger.info("MQTT connected rc=%s", rc)
    mqtt_client.on_disconnect = lambda c, u, rc: logger.warning("MQTT disconnected rc=%s", rc)
    try:
        mqtt_client.connect(broker, port, keepalive=60)
        mqtt_client.loop_start()
    except Exception:
        logger.exception("Failed to connect to MQTT broker %s:%s", broker, port)


def publish_alert(payload: Dict[str,Any]):
    if mqtt_client is None:
        offline_mqtt_queue.put(payload)
        logger.info("MQTT not ready, queued alert")
        return
    try:
        mqtt_client.publish(MQTT_TOPIC_ALERT, json.dumps(payload))
        logger.info("Published alert: %s", payload.get('intent'))
    except Exception:
        logger.exception("Publish failed, queueing")
        offline_mqtt_queue.put(payload)

# ---------------- TTS ----------------

tts_engine = pyttsx3.init()

def select_tts_voice(preferred_sequence=("en",)):
    try:
        voices = tts_engine.getProperty('voices')
        for pref in preferred_sequence:
            for v in voices:
                name = (getattr(v, 'name', '') or '').lower()
                id_ = (getattr(v, 'id', '') or '').lower()
                if pref in name or pref in id_:
                    tts_engine.setProperty('voice', v.id)
                    return v.id
    except Exception:
        logger.exception("Failed selecting TTS voice")
    return None

select_tts_voice(("en",))

def tts_worker():
    logger.info("TTS worker started")
    while running:
        try:
            text = tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if not text:
            continue
        if text == "__QUIT__":
            break
        try:
            logger.info("[TTS] %s", text)
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            logger.exception("TTS failed")
    logger.info("TTS worker exit")

# ---------------- Resemblyzer speaker enrollment & id ----------------

def init_resemblyzer():
    global encoder
    if not RESEMBLYZER_AVAILABLE:
        logger.warning("Resemblyzer not available; speaker features disabled")
        encoder = None
        return
    encoder = VoiceEncoder()


def enroll_speaker_resemblyzer(wav_path:str, name:str, out_dir: str = KNOWN_SPEAKERS_DIR):
    if encoder is None:
        logger.error("Encoder not initialised")
        return False
    try:
        cleaned = wav_path
        # Optional denoise
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            denoise_audio_file(wav_path, tmp.name, target_sr=DEFAULT_SAMPLE_RATE)
            wav = preprocess_wav(tmp.name)
        except Exception:
            wav = preprocess_wav(wav_path)
        emb = encoder.embed_utterance(wav)
        outp = os.path.join(out_dir, f"{name}.npy")
        np.save(outp, emb)
        logger.info("Saved speaker embedding %s", outp)
        return True
    except Exception:
        logger.exception("Enroll failed")
        return False


def identify_speaker_resemblyzer_from_bytes(wav_bytes:bytes):
    if encoder is None:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, np.frombuffer(wav_bytes, dtype='int16').astype('float32')/32767, DEFAULT_SAMPLE_RATE)
        wav = preprocess_wav(tmp.name)
        emb = encoder.embed_utterance(wav)
        best = None; best_sim = -1.0
        for f in glob.glob(os.path.join(KNOWN_SPEAKERS_DIR, '*.npy')):
            name = os.path.splitext(os.path.basename(f))[0]
            kvec = np.load(f)
            sim = float(np.dot(kvec, emb) / (np.linalg.norm(kvec)*np.linalg.norm(emb) + 1e-9))
            if sim > best_sim:
                best_sim = sim; best = name
        return (best, best_sim)
    except Exception:
        logger.exception("Speaker id failed")
        return None

# ---------------- NLU embedding-based classifier ----------------

def init_intent_model():
    global embed_model, intent_clf
    if not SKLEARN_AVAILABLE:
        logger.warning("Sentence-Transformers/Sklearn not available; using keyword NLU")
        embed_model = None
        intent_clf = None
        return
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # If you have a pre-trained classifier pickle, load here. Otherwise user can train with provided helper.
        intent_clf = None
    except Exception:
        logger.exception("Failed to init embed_model")
        embed_model = None
        intent_clf = None

TRAIN_EXAMPLES = [
    ("help me", "SOS"),("please help","SOS"),("i am trapped","SOS"),
    ("need water","NEED_SUPPLIES"),("need food","NEED_SUPPLIES"),
    ("fire nearby","HAZARD_FIRE"),("smoke","HAZARD_FIRE")
]
INTENT_MODEL_PATH = './intent_clf.pkl'

def train_intent_classifier(examples=TRAIN_EXAMPLES):
    global intent_clf, embed_model
    if embed_model is None:
        init_intent_model()
    if embed_model is None:
        logger.error("Cannot train intent model; embedding model missing")
        return
    texts = [t for t,_ in examples]
    labels = [l for _,l in examples]
    X = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, labels)
    intent_clf = clf
    import pickle
    with open(INTENT_MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    logger.info("Trained + saved intent classifier")


def load_intent_classifier():
    global intent_clf
    import pickle
    if os.path.exists(INTENT_MODEL_PATH):
        try:
            intent_clf = pickle.load(open(INTENT_MODEL_PATH, 'rb'))
            logger.info("Loaded intent classifier")
        except Exception:
            logger.exception("Failed to load intent classifier")


def predict_intent(text: str) -> str:
    t = text.lower()
    if intent_clf is not None and embed_model is not None:
        try:
            vec = embed_model.encode([text], convert_to_numpy=True)
            return intent_clf.predict(vec)[0]
        except Exception:
            pass
    # keyword fallback
    if any(k in t for k in ['sos','help','bachao','madad','trapped']):
        return 'SOS'
    if any(k in t for k in ['water','pani','drink']):
        return 'NEED_SUPPLIES'
    if any(k in t for k in ['fire','smoke']):
        return 'HAZARD_FIRE'
    return 'GENERAL'

# ---------------- ASR worker (multiprocessing) ----------------
# The worker process runs faster-whisper model and listens on a multiprocessing queue.

from multiprocessing import Process, Queue

def asr_worker_main(input_q: 'Queue', output_q: 'Queue', model_size:str, device:str, compute_type:str):
    # Run in child process
    if not WHISPER_AVAILABLE:
        logger.error("faster-whisper not available in ASR worker")
        return
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("ASR worker loaded model %s on %s", model_size, device)
    while True:
        job = input_q.get()
        if job is None:
            logger.info("ASR worker received shutdown")
            break
        wav_path = job.get('wav_path')
        meta = job.get('meta', {})
        try:
            # optionally denoise before transcribe
            transcribe_path = wav_path
            if NOISEREDUCE_AVAILABLE:
                tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                denoise_audio_file(wav_path, tmp.name, target_sr=16000)
                transcribe_path = tmp.name
            segments, info = model.transcribe(transcribe_path, beam_size=2, vad_filter=True)
            text = " ".join([s.text for s in segments]).strip()
            output_q.put({'text': text, 'meta': meta})
        except Exception:
            logger.exception("ASR worker failed on %s", wav_path)
            output_q.put({'text':'', 'meta': meta})

# ---------------- Audio capture & pipeline ----------------

vad = webrtcvad.Vad(VAD_MODE) if USE_VAD else None

def is_voice_frame(frame_bytes: bytes) -> bool:
    if vad is None:
        return True
    try:
        return vad.is_speech(frame_bytes, DEFAULT_SAMPLE_RATE)
    except Exception:
        return True

# helper to write buffer to wav file to pass to ASR worker
from scipy.io import wavfile

def write_wav_from_bytes(buf: bytes, path: str):
    arr = np.frombuffer(buf, dtype='int16')
    wavfile.write(path, DEFAULT_SAMPLE_RATE, arr)


def collect_and_queue_for_asr(stream, first_chunk: bytes, input_q: 'Queue'):
    # collect post buffer and write temp wav
    chunks = [first_chunk]
    end_time = time.time() + AUDIO_POST_SECONDS
    frame_size = int(DEFAULT_SAMPLE_RATE * 30 / 1000) * 2
    try:
        while time.time() < end_time:
            data, overflow = stream.read(frame_size)
            if not isinstance(data, (bytes, bytearray)):
                data = bytes(data)
            chunks.append(data)
    except Exception:
        logger.exception("Exception while collecting utterance")
    pre = b''.join(list(audio_ring))
    audio_bytes = pre + b''.join(chunks)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        write_wav_from_bytes(audio_bytes, tmp.name)
        input_q.put({'wav_path': tmp.name, 'meta': {}})
        logger.info("Queued audio for ASR worker: %s", tmp.name)
    except Exception:
        logger.exception("Failed to write wav for ASR")

# main capture loop puts files into input_q for ASR worker; reads results from output_q

def audio_capture_loop(input_q: 'Queue', output_q: 'Queue'):
    frame_duration = 30  # ms
    frame_size = int(DEFAULT_SAMPLE_RATE * frame_duration / 1000) * 2
    try:
        with sd.RawInputStream(samplerate=DEFAULT_SAMPLE_RATE, blocksize=frame_size, dtype='int16', channels=DEFAULT_CHANNELS) as stream:
            logger.info("Audio input stream opened")
            while running:
                data, overflow = stream.read(frame_size)
                if not isinstance(data, (bytes, bytearray)):
                    data = bytes(data)
                if overflow:
                    logger.warning("Audio overflow")
                audio_ring.append(data)

                if USE_VAD and not is_voice_frame(data):
                    continue

                # Wakeword detection by naive partial scan in the accumulated ring (cheap)
                if USE_WAKEWORD and WAKEWORD:
                    # quick check in last few chunks
                    joined = b"".join(list(audio_ring)[-10:])
                    # not reliable to search bytes for word; skip complex partial ASR here

                # If we detect energy/speech, collect a short utterance and queue
                # Heuristic: if vad says speech now, collect post buffer and send file to ASR worker
                if is_voice_frame(data):
                    collect_and_queue_for_asr(stream, data, input_q)

                # check for ASR outputs
                try:
                    while True:
                        out = output_q.get_nowait()
                        text = out.get('text','').strip()
                        if text:
                            handle_asr_text(text, out.get('meta', {}))
                except Exception:
                    pass
    except Exception:
        logger.exception("ASR audio stream failed")

# ---------------- Processing recognized text ----------------

def get_current_gps() -> Dict[str,float]:
    # TODO: integrate MAVLink telemetry
    return {"lat": 20.5937, "lon": 78.9629}


def get_drone_telemetry() -> Dict[str,Any]:
    return {"battery": None, "alt": None}


def handle_asr_text(text: str, meta: Dict[str,Any]):
    logger.info("Handle ASR text: %s", text)
    # speaker id attempt: we don't have spk vector from faster-whisper; we can run short diarized embedding if desired
    # For now, we can attempt to identify by re-playing the last saved temp wav if available in meta
    speaker = None
    speaker_sim = None
    if RESEMBLYZER_AVAILABLE:
        # if meta contains 'wav_path' or similar, identify
        wav_path = meta.get('wav_path') if meta else None
        if wav_path and os.path.exists(wav_path):
            try:
                wav = preprocess_wav(wav_path)
                emb = encoder.embed_utterance(wav)
                # compare with known
                best=None; best_sim=-1
                for f in glob.glob(os.path.join(KNOWN_SPEAKERS_DIR, '*.npy')):
                    name = os.path.splitext(os.path.basename(f))[0]
                    kvec = np.load(f)
                    sim = float(np.dot(kvec, emb) / (np.linalg.norm(kvec)*np.linalg.norm(emb)+1e-9))
                    if sim>best_sim: best_sim=sim; best=name
                speaker=best; speaker_sim=best_sim
            except Exception:
                logger.exception("Speaker id attempt failed")

    intent = predict_intent(text)
    logger.info("Intent: %s", intent)

    gps = get_current_gps()
    telemetry = get_drone_telemetry()
    payload = {
        'device_id': mqtt_client._client_id.decode() if mqtt_client else 'unknown',
        'intent': intent,
        'original_text': text,
        'english_text': text,
        'source_lang': 'en',
        'gps': gps,
        'telemetry': telemetry,
        'timestamp': int(time.time()),
        'meta': meta,
        'speaker': speaker,
        'speaker_sim': speaker_sim
    }

    if intent == 'SOS':
        payload['priority'] = 'high'

    publish_alert(payload)

    # TTS reply
    reply_en = "I hear you. Help is on the way."
    if intent == 'SOS':
        reply_en = "Help team alerted. If safe, move to open area and make noise every 30 seconds."
    elif intent == 'NEED_SUPPLIES':
        reply_en = "Noted. We'll arrange water, food and medical supplies. Any infants or elderly?"

    tts_queue.put(reply_en)

# ---------------- MAVLink hooks ----------------

def connect_mavlink(device: str, baud: int):
    if not MAVLINK_AVAILABLE:
        logger.info("pymavlink not available")
        return None
    try:
        m = mavutil.mavlink_connection(device, baud=baud)
        logger.info("MAVLink connected %s", device)
        return m
    except Exception:
        logger.exception("MAVLink connect failed")
        return None

# ---------------- Certificate helper (self-signed) ----------------

def generate_self_signed_cert(path_prefix: str, common_name: str = 'rescue-device') -> Tuple[str,str]:
    """
    Attempts to generate a self-signed certificate + key using openssl CLI.
    Produces path_prefix.crt and path_prefix.key. Returns (crt_path, key_path).
    """
    crt = f"{path_prefix}.crt"
    key = f"{path_prefix}.key"
    try:
        # openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key -out crt -subj "/CN=common_name"
        cmd = ["openssl","req","-x509","-nodes","-days","365","-newkey","rsa:2048","-keyout",key,"-out",crt,"-subj","/CN=%s" % common_name]
        subprocess.check_call(cmd)
        logger.info("Generated self-signed cert %s and key %s", crt, key)
        return (crt, key)
    except Exception:
        logger.exception("Failed to generate self-signed cert. Ensure openssl is installed and in PATH.")
        return (None, None)

# ---------------- MAIN & CLI ----------------

def graceful_shutdown(signum, frame):
    global running
    logger.info("Signal %s received, shutting down...", signum)
    running = False

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mqtt-broker', default=DEFAULT_MQTT_BROKER)
    parser.add_argument('--mqtt-port', type=int, default=DEFAULT_MQTT_PORT)
    parser.add_argument('--mqtt-user', default=None)
    parser.add_argument('--mqtt-pass', default=None)
    parser.add_argument('--tls-ca', default=None)
    parser.add_argument('--tls-cert', default=None)
    parser.add_argument('--tls-key', default=None)
    parser.add_argument('--asr-model', default=ASR_MODEL_SIZE)
    parser.add_argument('--asr-device', default=ASR_DEVICE)
    parser.add_argument('--asr-compute', default=ASR_COMPUTE_TYPE)
    parser.add_argument('--enroll', nargs=2, metavar=('WAV','NAME'), help='Enroll speaker from WAV')
    parser.add_argument('--gen-self-signed', nargs=1, metavar=('PREFIX',), help='Generate self-signed certs for testing')
    args = parser.parse_args()

    ensure_dirs()

    # generate certs if requested
    if args.gen_self_signed:
        prefix = args.gen_self_signed[0]
        generate_self_signed_cert(prefix)
        return

    # init resemblyzer
    init_resemblyzer()

    # Intent model
    init_intent_model()
    load_intent_classifier()

    # enroll flow
    if args.enroll:
        wav_path, name = args.enroll
        ok = enroll_speaker_resemblyzer(wav_path, name)
        if ok:
            logger.info("Enrollment successful for %s", name)
        else:
            logger.error("Enrollment failed for %s", name)
        return

    # mqtt
    mqtt_setup(args.mqtt_broker, args.mqtt_port, username=args.mqtt_user, password=args.mqtt_pass, tls_ca=args.tls_ca, tls_cert=args.tls_cert, tls_key=args.tls_key)

    # start TTS worker
    t_tts = threading.Thread(target=tts_worker, daemon=True)
    t_tts.start()

    # prepare multiprocessing ASR worker
    global asr_input_q, asr_output_q
    asr_input_q = Queue()
    asr_output_q = Queue()
    p = Process(target=asr_worker_main, args=(asr_input_q, asr_output_q, args.asr_model, args.asr_device, args.asr_compute), daemon=True)
    p.start()

    # audio capture loop runs in main thread and pushes files to asr_input_q
    try:
        audio_capture_loop(asr_input_q, asr_output_q)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in main")
    finally:
        logger.info("Shutting down: signalling ASR worker and threads")
        try:
            asr_input_q.put(None)
        except Exception:
            pass
        tts_queue.put('__QUIT__')
        time.sleep(0.5)
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()

if __name__ == '__main__':
    main()
