import os
import subprocess
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import base64
import math
import smtplib
import ssl
import requests
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

# Web server
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, send_from_directory

# Firebase
from firebase_admin import credentials, firestore
import firebase_admin

# Audio processing
import librosa
import pydub
from pydub import AudioSegment
import io

# Scheduler for automatic cleanup
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Data validation
from validators import (
    validate_email, validate_otp, validate_name, validate_password, validate_collection,
    validate_firestore_id, validate_alert_type, validate_video_filename, validate_user_data,
    validate_session_id, validate_cutoff_days, validate_manual_violation_type, validate_exam_data
)

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# Audio detection configuration - Device-specific thresholds
AUDIO_LOUD_THRESHOLD_DESKTOP = 0.75  # Desktop: RMS threshold for loud sound detection
AUDIO_LOUD_THRESHOLD_MOBILE = 0.45   # Mobile: Lower threshold (more sensitive) for better detection
                                      # Lower = more sensitive (detects quieter sounds)
                                      # Higher = less sensitive (only very loud sounds like shouting)
                                      # Desktop: 0.75 - detects normal talking and louder sounds
                                      # Mobile: 0.45 - detects quieter sounds for better mobile monitoring


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # Draw face landmarks as small filled dots (emphasize face mesh)
    if results.face_landmarks:
        h, w, _ = image.shape
        for lm in results.face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 1, (80, 110, 10), -1)

    # Draw left hand landmarks and connections
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

    # Draw right hand landmarks and connections
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def extract_keypoints(results):
    # keep helper if later needed; returns concatenated arrays (pose excluded to focus on face+hands)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([face, lh, rh])


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Dedicated local folder for all saved recordings
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), 'recordings')
WEB_RECORDINGS_PREFIX = '/recordings/'
# -----------------------------------------------
# FRONTEND ALERT → FIRESTORE (proctoring_sessions)
# -----------------------------------------------

# Map frontend types → Firestore flag keys used in your alertsSession.flags
FRONTEND_TYPE_MAP = {
    "keyboardShortcut": "keyboard_shortcut",
    "tabSwitch": "tab_switch",
    "rightClick": "right_click",
    "fullscreenExit": "fullscreen_exit",
    "captions": "captions",
}

@app.route('/frontend_alert', methods=['POST'])
def frontend_alert():
    """
    Receive cheating / behavior alerts from the frontend (take_exams.html)
    and append them into the proctoring_sessions.flags in Firestore.
    """
    try:
        data = request.get_json(force=True) or {}
        alert_type = data.get("type")
        user_id = data.get("user_id")
        session_id = data.get("session_id")

        valid, err = validate_alert_type(alert_type if alert_type else "")
        if not valid:
            return jsonify({"error": err}), 400
        valid, err = validate_session_id(session_id if session_id else "")
        if not valid:
            return jsonify({"error": err}), 400

        # Convert frontend type → flags key
        flag_key = FRONTEND_TYPE_MAP.get(alert_type)

        # Build the flag entry
        flag_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": alert_type,
            "user_id": user_id,
            "source": "frontend",
        }

        # If you pass extra fields (like details) from JS, keep them
        for k, v in data.items():
            if k not in ("type", "user_id", "session_id"):
                flag_entry[k] = v

        # Firestore client
        db = firestore.client()
        session_ref = db.collection("proctoring_sessions").document(session_id)

        # Make sure the session doc exists
        if not session_ref.get().exists:
            # Optional: create a basic session doc if not found
            session_ref.set({
                "user_id": user_id,
                "start_time": datetime.utcnow().isoformat() + "Z",
                "status": "active",
                "flags": {}
            }, merge=True)

        # Append this flag into flags.<flag_key> using ArrayUnion
        session_ref.update({
            f"flags.{flag_key}": firestore.ArrayUnion([flag_entry])
        })

        return jsonify({"status": "ok", "stored_as": flag_key}), 200

    except Exception as e:
        print("Error in /frontend_alert:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/add_manual_violation', methods=['POST'])
def add_manual_violation():
    """
    Add a manually tagged violation while reviewing session video.
    Expects JSON: session_id, type (violation type), message (optional), video_timestamp (optional, seconds into video).
    Appends to proctoring_sessions.flags with source: 'manual'.
    """
    try:
        data = request.get_json(force=True) or {}
        session_id = data.get('session_id')
        violation_type = data.get('type')
        message = (data.get('message') or '').strip()[:500]  # optional, max 500 chars
        video_timestamp = data.get('video_timestamp')  # optional, seconds into video

        valid, err = validate_session_id(session_id or '')
        if not valid:
            return jsonify({'error': err}), 400
        valid, err = validate_manual_violation_type(violation_type or '')
        if not valid:
            return jsonify({'error': err}), 400

        db_client = firestore.client()
        session_ref = db_client.collection('proctoring_sessions').document(session_id)
        session_doc = session_ref.get()
        if not session_doc.exists:
            return jsonify({'error': 'Session not found'}), 404

        flag_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': violation_type,
            'source': 'manual',
            'message': message or f'Manual tag: {violation_type}',
        }
        if video_timestamp is not None and isinstance(video_timestamp, (int, float)):
            flag_entry['video_timestamp'] = float(video_timestamp)

        # Map violation type to flags key (same key as auto-detected)
        flag_key = violation_type
        session_ref.update({
            f'flags.{flag_key}': firestore.ArrayUnion([flag_entry])
        })

        return jsonify({'status': 'ok', 'stored_as': flag_key}), 200
    except Exception as e:
        print('Error in /add_manual_violation:', e)
        return jsonify({'error': str(e)}), 500


## If you want to finally commit and push to git
# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("/etc/secrets/firebase_admin_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()


def count_faces_bounding_boxes(image, min_confidence=0.4):
    """Return number of faces detected and list of bounding boxes (x,y,w,h) in pixel coords."""
    # Use global face_detection_model (loaded once, not per request)
    global face_detection_model
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = face_detection_model.process(img_rgb)
    boxes = []
    if not res.detections:
        return 0, boxes
    h, w = image.shape[:2]
    for det in res.detections:
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        boxes.append((x, y, bw, bh))
    return len(boxes), boxes


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
YOLO_ONNX = os.path.join(MODEL_DIR, 'yolov5s.onnx')

# Load ML models once at startup (not per request) - CRITICAL for Render free tier memory optimization
yolo_net = None
if os.path.exists(YOLO_ONNX):
    try:
        yolo_net = cv2.dnn.readNet(YOLO_ONNX)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        yolo_net = None

# Initialize MediaPipe models once (not per request) - CRITICAL for memory optimization
# These are thread-safe and can be reused across requests
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False  # Video mode for better performance
)
face_detection_model = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 = short-range, 1 = full-range (0 is faster)
    min_detection_confidence=0.4
)
print("MediaPipe models initialized (loaded once, reused per request)")


def estimate_head_pose_from_mesh(results, image):
    """
    Improved head-pose estimate using 6 landmark correspondences and standard model_points.
    Returns (looking_bool, (yaw,pitch,roll)) or (False, None).
    """
    if not results.face_landmarks:
        return False, None

    # Use indices mapped to model_points order: nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
    indices = [1, 199, 33, 263, 61, 291]
    h, w = image.shape[:2]

    try:
        image_points = []
        for idx in indices:
            lm = results.face_landmarks.landmark[idx]
            image_points.append((lm.x * w, lm.y * h))
        image_points = np.array(image_points, dtype="double")

        # Standard 3D model points used in many head-pose examples (mm)
        model_points = np.array([
            (0.0, 0.0, 0.0),            # nose tip
            (0.0, -330.0, -65.0),      # chin
            (-225.0, 170.0, -135.0),   # left eye left corner
            (225.0, 170.0, -135.0),    # right eye right corner
            (-150.0, -150.0, -125.0),  # left mouth corner
            (150.0, -150.0, -125.0)    # right mouth corner
        ])

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return False, None

        rmat, _ = cv2.Rodrigues(rotation_vector)

        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0

        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)

        # Loosened thresholds (tune if needed). This is a robust heuristic.
        looking = (abs(yaw) < 30) and (abs(pitch) < 20)

        return looking, (float(yaw), float(pitch), float(roll))
    except Exception:
        return False, None


@app.route('/detect_audio', methods=['POST'])
def detect_audio():
    """
    Process audio data for loud sounds detection.
    Expects multipart/form-data with 'audio' file or base64 encoded audio in JSON/form.
    Can also accept 'rms_level' parameter directly for client-side calculated RMS.
    Query param: session_id (optional) for flagging
    Returns: {'loud_sound': bool, 'multiple_voices': bool, 'confidence': float}
    """
    try:
        session_id = request.args.get('session_id')
        user_id = request.args.get('user_id') or request.form.get('user_id')

        # Check if RMS level is provided directly (for client-side calculation)
        rms_level_param = request.form.get('rms_level')
        if rms_level_param:
            try:
                rms = float(rms_level_param)
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid rms_level parameter'}), 400
        else:
            # Get audio data for server-side calculation
            audio_data = None
            if 'audio' in request.files:
                audio_file = request.files['audio']
                audio_data = audio_file.read()
            elif request.is_json:
                data = request.get_json()
                b64_audio = data.get('audio') or data.get('audio_b64')
                if b64_audio:
                    if b64_audio.startswith('data:'):
                        _, b64_audio = b64_audio.split(',', 1)
                    audio_data = base64.b64decode(b64_audio)
            elif request.form:
                b64_audio = request.form.get('audio') or request.form.get('audio_b64')
                if b64_audio:
                    if b64_audio.startswith('data:'):
                        _, b64_audio = b64_audio.split(',', 1)
                    audio_data = base64.b64decode(b64_audio)

            if not audio_data:
                return jsonify({'error': 'No audio data or rms_level provided'}), 400

            # Convert audio data to numpy array
            try:
                # Try to load as WAV/PCM first
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
                samples = np.array(audio_segment.get_array_of_samples())
                if audio_segment.channels == 2:
                    samples = samples.reshape((-1, 2))
                sample_rate = audio_segment.frame_rate
            except Exception:
                # Fallback: assume raw PCM data
                samples = np.frombuffer(audio_data, dtype=np.int16)
                sample_rate = 44100  # default assumption

            # Convert to mono if stereo
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)

            # Convert to float
            samples = samples.astype(np.float32)
            if samples.max() > 1.0:
                samples = samples / 32768.0  # normalize 16-bit

            # Detect loud sounds (simple RMS threshold)
            rms = np.sqrt(np.mean(samples**2))

        # Check if request is from mobile device
        user_agent = request.headers.get('User-Agent', '').lower()
        is_mobile = any(keyword in user_agent for keyword in [
            'android', 'webos', 'iphone', 'ipad', 'ipod', 'blackberry', 'iemobile', 'opera mini'
        ])

        # Use appropriate threshold based on device type
        threshold = AUDIO_LOUD_THRESHOLD_MOBILE if is_mobile else AUDIO_LOUD_THRESHOLD_DESKTOP
        is_loud = rms > threshold

        print(f"Audio detection - RMS: {rms:.3f}, Threshold: {threshold:.3f}, Mobile: {is_mobile}, Loud: {is_loud}")

        # Detect loud sounds only (removed multiple voices detection)
        # Flag session if session_id provided
        if session_id:
            flags = {}
            if is_loud:
                flags['loud_sound'] = [{
                    'timestamp': datetime.utcnow().isoformat(),
                    'message': f'Loud sound detected (RMS: {rms:.3f})'
                }]

            if flags:
                try:
                    # Get current session data
                    session_ref = db.collection('proctoring_sessions').document(session_id)
                    session_doc = session_ref.get()
                    if not session_doc.exists:
                        session_ref.set({
                            'user_id': user_id,
                            'start_time': datetime.utcnow().isoformat() + "Z",
                            'status': 'active',
                            'flags': {}
                        }, merge=True)
                    current_flags = session_doc.to_dict().get('flags', {}) if session_doc.exists else {}
                    # Merge new flags with existing flags
                    for key, value in flags.items():
                        if key not in current_flags:
                            current_flags[key] = []
                        current_flags[key].extend(value)
                    session_ref.update({'flags': current_flags})
                except Exception as e:
                    print('Warning: failed to update session flags:', e)

        return jsonify({
            'loud_sound': bool(is_loud),
            'rms_level': float(rms)
        })

    except Exception as e:
        print('Audio detection error:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Process video frame for cheating detection.
    
    NOTE: For Render free tier hosting:
    - This endpoint is CPU-intensive (MediaPipe, YOLO, OpenCV)
    - Frontend is configured to send frames every 5-10 seconds (not every frame)
    - Consider reducing processing frequency or using client-side detection only
    - Free tier has 30-second timeout limits - processing should complete quickly
    """
    session_id = request.args.get('session_id')
    user_id = request.form.get('user_id') or request.args.get('user_id')

    file = None
    for key in ('frame', 'file', 'image'):
        if key in request.files:
            file = request.files[key]
            break

    img = None
    if file:
        data = file.read()
        if not data:
            return jsonify({'error': 'uploaded file empty'}), 400
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'cannot decode uploaded file'}), 400

    # 2) Try form field containing data URI or base64
    if img is None and request.form:
        val = request.form.get('frame') or request.form.get('image') or request.form.get('file')
        if val:
            if val.startswith('data:'):
                try:
                    header, b64 = val.split(',', 1)
                    data = base64.b64decode(b64)
                    nparr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    return jsonify({'error': 'invalid data URI in form field'}), 400
            else:
                try:
                    data = base64.b64decode(val)
                    nparr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    return jsonify({'error': 'invalid base64 in form field'}), 400

    # 3) Try JSON payload
    if img is None and request.is_json:
        j = request.get_json(silent=True) or {}
        b64 = j.get('image') or j.get('image_b64') or j.get('frame')
        if b64:
            if b64.startswith('data:'):
                _, b64 = b64.split(',', 1)
            try:
                data = base64.b64decode(b64)
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                return jsonify({'error': 'invalid base64 in JSON'}), 400

    if img is None and request.data:
        raw = request.data
        # if starts with data URI text, handle
        try:
            s = raw.decode('utf-8', errors='ignore')
            if s.startswith('data:'):
                _, b64 = s.split(',', 1)
                raw = base64.b64decode(b64)
        except Exception:
            pass
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'no image found in request. send multipart/form-data with field "frame" or JSON/form with base64 data.'}), 400

    frame = img  # BGR image for further processing

    # 1) count faces using FaceDetection (can find multiple)
    face_count, face_boxes = count_faces_bounding_boxes(frame, min_confidence=0.4)

    # 2) process with Mediapipe face mesh / hands via Holistic to draw landmarks and estimate pose
    # Use global holistic_model (loaded once, not per request) - CRITICAL for memory optimization
    global holistic_model
    image, results = mediapipe_detection(frame, holistic_model)
    draw_styled_landmarks(image, results)

    # estimate head pose / gaze
    looking, angles = estimate_head_pose_from_mesh(results, image)

    # Extract keypoints for potential ML model integration
    keypoints = extract_keypoints(results)

    # 3) Object detection using YOLO for phones and other objects
    # Resize frame to 640x640 for YOLO (reduces memory usage)
    # Original frame is kept for display, but YOLO processes smaller version
    detected_objects = []
    if yolo_net is not None:
        try:
            # Resize frame to 640x640 for YOLO processing (memory optimization)
            h, w = frame.shape[:2]
            if h > 640 or w > 640:
                scale = min(640 / h, 640 / w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame
                new_w, new_h = w, h
            
            blob = cv2.dnn.blobFromImage(frame_resized, 1 / 255.0, (640, 640), swapRB=True, crop=False)
            yolo_net.setInput(blob)
            outputs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())

            # Use resized dimensions for YOLO output, then scale back to original
            yolo_h, yolo_w = 640, 640
            scale_x = w / yolo_w
            scale_y = h / yolo_h
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # confidence threshold
                        center_x = int(detection[0] * yolo_w * scale_x)
                        center_y = int(detection[1] * yolo_h * scale_y)
                        width = int(detection[2] * yolo_w * scale_x)
                        height = int(detection[3] * yolo_h * scale_y)
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)

                        # COCO class names (assuming standard YOLOv5 COCO model)
                        class_names = [
                            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                            'toothbrush'
                        ]

                        if class_id < len(class_names):
                            obj_name = class_names[class_id]
                            detected_objects.append({
                                'name': obj_name,
                                'confidence': float(confidence),
                                'bbox': [x, y, width, height]
                            })

                            # Draw bounding box for detected objects
                            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
                            cv2.putText(
                                image,
                                f"{obj_name}: {confidence:.2f}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2
                            )
        except Exception as e:
            print(f"YOLO detection error: {e}")
            detected_objects = []
    else:
        detected_objects = []

    # overlay hints and boxes
    if face_count:
        for (x, y, wc, hc) in face_boxes:
            cv2.rectangle(image, (x, y), (x + wc, y + hc), (0, 200, 0), 2)
    cv2.putText(
        image,
        f"Faces: {face_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 255, 200),
        2,
        cv2.LINE_AA
    )

    # encode to JPEG and base64
    ret, buffer = cv2.imencode('.jpg', image)
    if not ret:
        return jsonify({'error': 'encoding failed'}), 500
    jpg_bytes = buffer.tobytes()
    b64 = base64.b64encode(jpg_bytes).decode('utf-8')
    data_uri = f'data:image/jpeg;base64,{b64}'

    # Flag session if session_id provided
    if session_id:
        try:
            flags = {}
            if face_count > 1:
                flags['twofaces'] = [{
                    'timestamp': datetime.utcnow().isoformat(),
                    'message': f'Multiple faces detected ({face_count})'
                }]

            # Add object detection flags
            for obj in detected_objects:
                if obj['name'] in ['cell phone', 'laptop', 'book', 'remote', 'keyboard']:
                    flag_key = f'object_detected_{obj["name"].replace(" ", "_")}'
                    flags[flag_key] = [{
                        'timestamp': datetime.utcnow().isoformat(),
                        'message': f'{obj["name"]} detected with confidence {obj["confidence"]:.2f}'
                    }]

            if flags:
                # Update proctoring_sessions with flags
                session_ref = db.collection('proctoring_sessions').document(session_id)
                if not session_ref.get().exists:
                    session_ref.set({
                        'user_id': user_id,
                        'exam_id': request.args.get('exam_id'),  # if passed
                        'start_time': datetime.utcnow().isoformat() + "Z",
                        'status': 'active',
                        'flags': {}
                    }, merge=True)
                session_ref.update({
                    'flags': flags
                })
        except Exception as e:
            print('Warning: failed to update session flags:', e)

    # return detection metadata
    return jsonify({
        'processed_image': data_uri,
        'label': None,
        'probability': 0.0,
        'face_count': face_count,
        'looking_at_screen': bool(looking),
        'head_angles': {
            'yaw': angles[0],
            'pitch': angles[1],
            'roll': angles[2]
        } if angles else None,
        'detected_objects': detected_objects,
        'keypoints': keypoints.tolist() if keypoints is not None else None,  # Include keypoints in response
    })


# Update the existing CRUD routes to handle different collections

@app.route('/add', methods=['POST'])
def add():
    try:
        collection = request.args.get('collection', 'users')
        valid, err = validate_collection(collection)
        if not valid:
            return jsonify({'error': err}), 400
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object'}), 400
        if collection == 'users':
            valid, err = validate_user_data(data)
            if not valid:
                return jsonify({'error': err}), 400
        elif collection == 'exams':
            valid, err = validate_exam_data(data, require_all_fields=True)
            if not valid:
                return jsonify({'error': err}), 400
        doc_ref = db.collection(collection).add(data)
        return jsonify({'id': doc_ref[1].id}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/select')
def select():
    try:
        collection = request.args.get('collection', 'users')
        valid, err = validate_collection(collection)
        if not valid:
            return jsonify({'error': err}), 400
        doc_id = request.args.get('id')

        if doc_id:
            # Get single document
            doc = db.collection(collection).document(doc_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return jsonify(data), 200
            return jsonify({'error': 'Document not found'}), 404

        # Get all documents
        docs = []
        for doc in db.collection(collection).stream():
            data = doc.to_dict()
            data['id'] = doc.id
            docs.append(data)
        return jsonify(docs), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/edit/<id>', methods=['POST'])
def edit(id):
    try:
        collection = request.args.get('collection', 'users')
        valid, err = validate_collection(collection)
        if not valid:
            return jsonify({'error': err}), 400
        valid, err = validate_firestore_id(id, 'id')
        if not valid:
            return jsonify({'error': err}), 400
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object'}), 400
        if collection == 'exams':
            valid, err = validate_exam_data(data, require_all_fields=False)
            if not valid:
                return jsonify({'error': err}), 400
        db.collection(collection).document(id).update(data)
        return jsonify({'message': 'Updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete/<id>', methods=['POST'])
def delete(id):
    try:
        collection = request.args.get('collection', 'users')
        valid, err = validate_collection(collection)
        if not valid:
            return jsonify({'error': err}), 400
        valid, err = validate_firestore_id(id, 'id')
        if not valid:
            return jsonify({'error': err}), 400
        db.collection(collection).document(id).delete()
        return jsonify({'message': 'Deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    users_ref = db.collection('users')
    query = users_ref.where('email', '==', email).where('password', '==', password).limit(1)
    user = next(query.stream(), None)

    if user:
        user_data = user.to_dict()
        user_data['id'] = user.id  # Add document ID to user data
        return jsonify({
            'success': True,
            'user': user_data
        }), 200

    return jsonify({
        'success': False,
        'error': 'Invalid credentials'
    }), 401


@app.route('/')
def login_page():
    return render_template('login.html')


@app.route('/detection')
def detection():
    return render_template('index.html')


# OTP store: {email: {"otp": str, "expires": datetime}}
_otp_store = {}
OTP_EXPIRY_MINUTES = 5
OTP_LENGTH = 6


def _sendgrid_send_email(subject: str, to_email: str, text_body: str) -> bool:
    """
    Send a plain-text email via SendGrid Web API.

    Env vars:
      - SENDGRID_API_KEY (required)
      - EMAIL_FROM (required)  # must be a verified sender in SendGrid
      - EMAIL_TIMEOUT (optional, seconds; default 20)
    """
    api_key = os.environ.get('SENDGRID_API_KEY')
    from_email = os.environ.get('EMAIL_FROM')
    timeout_s = int(os.environ.get('EMAIL_TIMEOUT', '20'))
    if not api_key:
        print("[SendGrid] Not configured (missing SENDGRID_API_KEY).")
        return False
    if not from_email:
        print("[SendGrid] Not configured (missing EMAIL_FROM).")
        return False

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/plain", "value": text_body}],
    }
    try:
        res = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_s,
        )
        if 200 <= res.status_code < 300:
            return True
        # SendGrid often includes useful error details in JSON.
        print(f"[SendGrid] Send failed to {to_email}: HTTP {res.status_code} {res.text}")
        return False
    except Exception as e:
        print(f"[SendGrid] Send failed to {to_email}: {e}")
        return False


def _smtp_send_email(subject: str, to_email: str, text_body: str) -> bool:
    """
    Send a plain-text email via SMTP.

    Env vars:
      - SMTP_HOST (required)
      - SMTP_PORT (default 587)
      - SMTP_USER (required)
      - SMTP_PASS (required)  # For Gmail, use an App Password (requires 2FA)
      - SMTP_FROM (optional, defaults to SMTP_USER)
      - SMTP_USE_SSL (optional: "1"/"true" to force SMTP_SSL; auto-true for port 465)
      - SMTP_TIMEOUT (optional, seconds; default 20)
    """
    host = os.environ.get('SMTP_HOST')
    port = int(os.environ.get('SMTP_PORT', '587'))
    user = os.environ.get('SMTP_USER')
    password = os.environ.get('SMTP_PASS')
    from_email = os.environ.get('SMTP_FROM') or user
    timeout_s = int(os.environ.get('SMTP_TIMEOUT', '20'))
    use_ssl_env = (os.environ.get('SMTP_USE_SSL') or '').strip().lower()
    use_ssl = (port == 465) or use_ssl_env in {'1', 'true', 'yes', 'on'}

    if not host or not user or not password:
        print(f"[SMTP] Not configured (missing SMTP_HOST/SMTP_USER/SMTP_PASS). Would send to {to_email}.")
        return True

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    msg.attach(MIMEText(text_body, 'plain'))

    try:
        context = ssl.create_default_context()
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=timeout_s, context=context) as s:
                s.ehlo()
                s.login(user, password)
                s.sendmail(from_email, to_email, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=timeout_s) as s:
                s.ehlo()
                s.starttls(context=context)
                s.ehlo()
                s.login(user, password)
                s.sendmail(from_email, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[SMTP] Send failed to {to_email}: {e}")
        return False


def _send_email(subject: str, to_email: str, text_body: str) -> bool:
    """
    Send an email using the best available provider.
    Priority:
      1) SendGrid (if SENDGRID_API_KEY is set)
      2) SMTP (if SMTP_* is set)
      3) If neither is configured, log OTP to console and return True (dev mode)
    """
    if os.environ.get('SENDGRID_API_KEY'):
        return _sendgrid_send_email(subject, to_email, text_body)

    host = os.environ.get('SMTP_HOST')
    user = os.environ.get('SMTP_USER')
    password = os.environ.get('SMTP_PASS')
    if host and user and password:
        return _smtp_send_email(subject, to_email, text_body)

    print(f"[Email] No email provider configured. Would send to {to_email}. Subject: {subject}. Body: {text_body}")
    return True


def _send_otp_email(to_email: str, otp: str) -> bool:
    """Send OTP via configured email provider. Returns True on success."""
    text = f'Your verification code is: {otp}\n\nValid for {OTP_EXPIRY_MINUTES} minutes.'
    ok = _send_email('Your verification code', to_email, text)
    if not ok:
        print("[OTP] Email send failed (see provider log above).")
    return ok


@app.route('/send_otp', methods=['POST'])
def send_otp():
    """Generate OTP, store it, and send via email. Expects JSON: {email: str}."""
    try:
        data = request.get_json() or {}
        email = (data.get('email') or '').strip().lower()
        valid, err = validate_email(email)
        if not valid:
            return jsonify({'error': err}), 400
        otp = ''.join(secrets.choice('0123456789') for _ in range(OTP_LENGTH))
        expires = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        _otp_store[email] = {'otp': otp, 'expires': expires}
        if not _send_otp_email(email, otp):
            return jsonify({'error': 'Failed to send email'}), 500
        return jsonify({'status': 'ok', 'message': 'OTP sent to your email'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    """Verify OTP for an email. Expects JSON: {email: str, otp: str}. Returns 200 if valid."""
    try:
        data = request.get_json() or {}
        email = (data.get('email') or '').strip().lower()
        otp = (data.get('otp') or '').strip()
        valid, err = validate_email(email)
        if not valid:
            return jsonify({'valid': False, 'error': err}), 400
        valid, err = validate_otp(otp)
        if not valid:
            return jsonify({'valid': False, 'error': err}), 400
        stored = _otp_store.get(email)
        if not stored:
            return jsonify({'valid': False, 'error': 'No OTP found. Request a new code.'}), 400
        if datetime.utcnow() > stored['expires']:
            del _otp_store[email]
            return jsonify({'valid': False, 'error': 'OTP expired. Request a new code.'}), 400
        if stored['otp'] != otp:
            return jsonify({'valid': False, 'error': 'Invalid OTP'}), 400
        del _otp_store[email]
        return jsonify({'valid': True, 'status': 'ok'}), 200
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 500


PHONE_VIOLATION_FLAGS = {'phone', 'object_detected_cell_phone'}


def _send_violation_email_to_program_chair(to_email: str, student_name: str, student_email: str,
                                           exam_title: str, session_date: str, session_id: str) -> bool:
    """Send academic integrity incident email to program chair. Returns True on success."""
    try:
        subject = 'Academic Integrity Incident – Phone Violation During Proctored Exam'
        body = f"""Dear Program Chair,

This email reports an academic integrity incident that occurred during a proctored exam.

**Student Information**
• Name: {student_name}
• Email: {student_email}
• Course/Exam: {exam_title}
• Date & Time: {session_date}

**Violation Details**
During the exam, a cell phone was detected. The exam was terminated per academic integrity policy, which requires immediate termination for phone violations.

**Session ID:** {session_id}
(Use this ID to review the session recording and evidence in the Proctoring Reports.)

Please advise on next steps per your academic integrity procedures.

Best regards,
Proctoring System"""
        ok = _send_email(subject, to_email, body)
        if not ok:
            print(f"[Program Chair] Email send failed for session {session_id} (see provider log above).")
        return ok
    except Exception as e:
        print(f"[Program Chair] Email send failed: {e}")
        return False


@app.route('/notify_program_chair', methods=['POST'])
def notify_program_chair():
    """
    Send email to program chair when a session has a phone violation.
    Expects JSON: session_id, program_chair_email.
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id') or ''
        program_chair_email = (data.get('program_chair_email') or '').strip().lower()
        valid, err = validate_session_id(session_id)
        if not valid:
            return jsonify({'error': err}), 400
        valid, err = validate_email(program_chair_email)
        if not valid:
            return jsonify({'error': err or 'Invalid program chair email'}), 400

        db_client = firestore.client()
        session_ref = db_client.collection('proctoring_sessions').document(session_id)
        session_doc = session_ref.get()
        if not session_doc.exists:
            return jsonify({'error': 'Session not found'}), 404
        session = session_doc.to_dict()
        session['id'] = session_doc.id
        flags = session.get('flags') or {}

        has_phone = any(
            (flags.get(k) or []) for k in PHONE_VIOLATION_FLAGS
        )
        if not has_phone:
            return jsonify({'error': 'No phone violation found in this session'}), 400

        student_id = session.get('student_id') or session.get('user_id') or ''
        exam_id = session.get('exam_id') or ''
        student_name = 'Unknown'
        student_email = ''
        exam_title = 'Unknown Exam'
        if student_id:
            user_doc = db_client.collection('users').document(student_id).get()
            if user_doc.exists:
                u = user_doc.to_dict()
                student_name = u.get('name') or u.get('email') or student_id
                student_email = u.get('email') or ''
        if exam_id:
            exam_doc = db_client.collection('exams').document(exam_id).get()
            if exam_doc.exists:
                exam_title = exam_doc.to_dict().get('title') or exam_title

        raw_start = session.get('start_time')
        try:
            if hasattr(raw_start, 'strftime'):
                session_date = raw_start.strftime('%Y-%m-%d %H:%M') if raw_start else 'N/A'
            elif raw_start:
                dt = datetime.fromisoformat(str(raw_start).replace('Z', '+00:00'))
                session_date = dt.strftime('%Y-%m-%d %H:%M')
            else:
                session_date = 'N/A'
        except Exception:
            session_date = str(raw_start)[:19] if raw_start else 'N/A'

        if not _send_violation_email_to_program_chair(
            program_chair_email, student_name, student_email, exam_title, session_date, session_id
        ):
            return jsonify({'error': 'Failed to send email. Check SMTP configuration.'}), 500

        return jsonify({'status': 'ok', 'message': 'Email sent to program chair'}), 200
    except Exception as e:
        print('Error in /notify_program_chair:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/user_guide')
def user_guide():
    """Public user guide with Student and Proctor dashboard instructions."""
    return render_template('user_guide.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/student_dashboard')
def student_dashboard():
    return render_template('student_dashboard.html')


@app.route('/users')
def users():
    return render_template('users.html')


@app.route('/exams')
def exams():
    return render_template('exams.html')


@app.route('/classes')
def classes():
    return render_template('classes.html')


@app.route('/join/<code>')
def join_class(code):
    """Invite link: teacher shares this URL; students click to join the class (like Google Classroom)."""
    return render_template('join.html')


@app.route('/reports')
def reports():
    return render_template('reports.html')


@app.route('/view_recording')
def view_recording():
    """Standalone page to view session recording - use Chrome/Edge for best playback."""
    return render_template('view_recording.html')


@app.route('/take_exam/<exam_id>')
def take_exam(exam_id):
    exam_ref = db.collection('exams').document(exam_id)
    exam = exam_ref.get()

    if not exam.exists:
        flash('Exam not found.', 'error')
        return redirect('/student_dashboard')

    exam_data = exam.to_dict()

    # Check deadlines
    now = datetime.utcnow()
    if exam_data.get('start_date') and exam_data.get('end_date'):
        start_date = datetime.fromisoformat(exam_data['start_date'])
        end_date = datetime.fromisoformat(exam_data['end_date'])

        if now < start_date:
            flash('This exam has not started yet.', 'warning')
            return redirect('/student_dashboard')
        elif now > end_date:
            flash('This exam has expired and is no longer available.', 'error')
            return redirect('/student_dashboard')

    # Validate student section (if user_id is provided via query parameter)
    # Note: In production, use proper session management instead of query parameters
    user_id = request.args.get('user_id')
    if user_id and exam_data.get('class_id'):
        try:
            user_ref = db.collection('users').document(user_id)
            user = user_ref.get()
            if user.exists:
                user_data = user.to_dict()
                student_class_id = user_data.get('class_id')
                
                # Check if student's class matches exam's class
                if student_class_id and student_class_id != exam_data.get('class_id'):
                    flash('You are not authorized to take this exam. This exam is for a different section.', 'error')
                    return redirect('/student_dashboard')
        except Exception as e:
            print(f'Error validating student section: {e}')
            # Continue anyway if validation fails (for backward compatibility)

    return render_template('index.html', exam_id=exam_id, user_id=user_id)


@app.route('/profile')
def profile():
    return render_template('profile.html')


@app.route('/proctor_dashboard')
def proctor_dashboard():
    return render_template('proctor_dashboard.html')


@app.route('/logout', methods=['POST'])
def logout():
    return jsonify({'message': 'Logged out successfully'}), 200


@app.route('/delete_old_reports', methods=['POST'])
def delete_old_reports():
    """
    Manually delete old proctoring reports older than specified date.
    Expects JSON with 'cutoff_date' (ISO format) or 'cutoff_days' (default: 14)
    Requires admin authentication.
    """
    try:
        data = request.get_json() or {}
        delete_all = data.get('delete_all', False)

        if delete_all:
            print("Starting manual cleanup of ALL reports...")
            cutoff_iso = None  # Delete all, no cutoff
        else:
            cutoff_date_str = data.get('cutoff_date')
            cutoff_days = data.get('cutoff_days', 14)

            if cutoff_date_str:
                # Parse specific date
                try:
                    cutoff_date = datetime.fromisoformat(cutoff_date_str.replace('Z', '+00:00'))
                    cutoff_iso = cutoff_date.isoformat()
                    print(f"Starting manual cleanup of reports older than {cutoff_date_str}...")
                except (ValueError, TypeError):
                    return jsonify({'error': 'cutoff_date must be a valid ISO format date'}), 400
            else:
                # Use days (backward compatibility)
                try:
                    cutoff_days = int(cutoff_days)
                except (ValueError, TypeError):
                    return jsonify({'error': 'cutoff_days must be a valid integer'}), 400
                valid, err = validate_cutoff_days(cutoff_days)
                if not valid:
                    return jsonify({'error': err}), 400

                cutoff_date = datetime.utcnow() - timedelta(days=cutoff_days)
                cutoff_iso = cutoff_date.isoformat()
                print(f"Starting manual cleanup of reports older than {cutoff_days} days...")

        # Delete old proctoring sessions
        sessions_ref = db.collection('proctoring_sessions')
        if cutoff_iso is None:
            # Delete ALL sessions
            old_sessions = sessions_ref.stream()
        else:
            old_sessions = sessions_ref.where('start_time', '<', cutoff_iso).stream()

        deleted_count = 0
        for session_doc in old_sessions:
            session_data = session_doc.to_dict()

            # Delete associated video files (webcam, clips, and screen recordings)
            for path_key in ['video_path', 'clip_path', 'screen_path']:
                if session_data.get(path_key):
                    # Convert web path to file system path (support both /recordings/ and legacy /static/recordings/)
                    web_path = session_data[path_key]
                    if web_path.startswith(WEB_RECORDINGS_PREFIX):
                        filename = web_path.replace(WEB_RECORDINGS_PREFIX, '')
                        file_path = os.path.join(RECORDINGS_DIR, filename)
                    elif web_path.startswith('/static/recordings/'):
                        filename = web_path.replace('/static/recordings/', '')
                        file_path = os.path.join(os.path.dirname(__file__), 'static', 'recordings', filename)
                    else:
                        file_path = None
                    if file_path:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {e}")

            # Delete the session document
            session_doc.reference.delete()
            deleted_count += 1

        print(f"Manually deleted {deleted_count} old proctoring sessions")

        # If deleting all, also delete exam_results to clear the student scores table
        if cutoff_iso is None:
            results_ref = db.collection('exam_results')
            deleted_results = 0
            for result_doc in results_ref.stream():
                result_doc.reference.delete()
                deleted_results += 1
            print(f"Deleted {deleted_results} exam results")

        # Clean up expired shared reports (if any have expiration set)
        shared_ref = db.collection('shared_reports')
        expired_shares = shared_ref.where('expires_at', '<', datetime.utcnow().isoformat()).stream()

        expired_count = 0
        for share_doc in expired_shares:
            share_doc.reference.delete()
            expired_count += 1

        print(f"Deleted {expired_count} expired shared reports")

        return jsonify({
            'success': True,
            'message': f'Successfully deleted {deleted_count} old reports and {expired_count} expired shares'
        }), 200

    except Exception as e:
        print(f"Error during manual cleanup: {e}")
        return jsonify({'error': str(e)}), 500


def _convert_webm_to_mp4(webm_path):
    """Convert WebM to MP4 using ffmpeg. Returns (mp4_path, None) on success or (None, error_message) on failure."""
    if not os.path.isfile(webm_path):
        return None, "File not found"
    size = os.path.getsize(webm_path)
    if size == 0:
        return None, "Recording file is empty (0 bytes)"
    fd, mp4_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    try:
        # Lenient flags to salvage browser-recorded WebM (incomplete/corrupted)
        cmd = [
            'ffmpeg', '-y',
            '-fflags', '+genpts+igndts',
            '-err_detect', 'ignore_err',
            '-i', webm_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-max_muxing_queue_size', '1024',
            mp4_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "Unknown error")[-500:]
            print(f"ffmpeg conversion failed: {err}")
            try:
                os.unlink(mp4_path)
            except OSError:
                pass
            last_line = err.strip().split('\n')[-1] if err.strip() else "check console"
            return None, f"ffmpeg failed: {last_line}"
        if not os.path.isfile(mp4_path) or os.path.getsize(mp4_path) == 0:
            try:
                os.unlink(mp4_path)
            except OSError:
                pass
            return None, "Conversion produced empty file"
        return mp4_path, None
    except FileNotFoundError:
        print("ffmpeg not found. Install from https://ffmpeg.org/download.html and add to PATH.")
        try:
            os.unlink(mp4_path)
        except OSError:
            pass
        return None, "ffmpeg not found. Install ffmpeg and add its bin folder to PATH."
    except subprocess.TimeoutExpired:
        try:
            os.unlink(mp4_path)
        except OSError:
            pass
        return None, "Conversion timed out"
    except OSError as e:
        try:
            os.unlink(mp4_path)
        except OSError:
            pass
        return None, str(e)


@app.route('/recordings/<path:filename>')
def serve_recording(filename):
    """Serve recorded video files from the local recordings folder.
    Add ?download=1 to force download (save to computer) instead of inline playback.
    Add ?format=mp4 to convert WebM to MP4 before serving (requires ffmpeg)."""
    webm_path = os.path.join(RECORDINGS_DIR, filename)
    as_attachment = request.args.get('download') == '1'
    convert_mp4 = request.args.get('format') == 'mp4'

    if convert_mp4 and filename.lower().endswith('.webm'):
        mp4_path, err_msg = _convert_webm_to_mp4(webm_path)
        if mp4_path:
            try:
                download_name = os.path.splitext(filename)[0] + '.mp4'
                return send_from_directory(
                    os.path.dirname(mp4_path),
                    os.path.basename(mp4_path),
                    as_attachment=True,
                    download_name=download_name
                )
            finally:
                try:
                    os.unlink(mp4_path)
                except OSError:
                    pass
        # Conversion failed: return error so user sees message instead of wrong file
        return jsonify({'error': err_msg or 'MP4 conversion failed'}), 503

    file_path = os.path.join(RECORDINGS_DIR, filename)
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return jsonify({'error': 'Recording file not found or empty'}), 404

    mimetype = None
    if filename.lower().endswith('.webm'):
        mimetype = 'video/webm'
    elif filename.lower().endswith('.mp4'):
        mimetype = 'video/mp4'
    return send_from_directory(RECORDINGS_DIR, filename, as_attachment=as_attachment, mimetype=mimetype)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Accepts multipart/form-data video upload and attaches it to a proctoring session.
    Query param: session_id (required) or form field session_id
    Query param: type (optional, 'clip' for cheating clips)
    Saves file to local recordings/ folder and updates proctoring_sessions document with video_path or clip_path.
    """
    try:
        session_id = request.args.get('session_id') or request.form.get('session_id')
        upload_type = request.args.get('type', 'full')  # 'full', 'clip', or 'screen'
        valid, err = validate_session_id(session_id or '')
        if not valid:
            return jsonify({'error': err}), 400

        if 'video' not in request.files:
            return jsonify({'error': 'no video file sent'}), 400

        f = request.files['video']
        if f.filename == '':
            return jsonify({'error': 'empty filename'}), 400

        valid, err = validate_video_filename(f.filename)
        if not valid:
            return jsonify({'error': err}), 400

        safe_name = secure_filename(f.filename)
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        filename = f"session_{session_id}_{timestamp}_{safe_name}"
        save_path = os.path.join(RECORDINGS_DIR, filename)
        f.save(save_path)

        size = os.path.getsize(save_path)
        if size == 0:
            try:
                os.unlink(save_path)
            except OSError:
                pass
            return jsonify({'error': 'Recording is empty (0 bytes). Use Chrome, ensure camera/mic work, and let recording run 10+ seconds before submitting.'}), 400

        web_path = f"{WEB_RECORDINGS_PREFIX}{filename}"

        # update proctoring_sessions document
        try:
            if upload_type == 'clip':
                db.collection('proctoring_sessions').document(session_id).update({
                    'clip_path': web_path,
                    'clip_uploaded_at': datetime.utcnow().isoformat()
                })
            elif upload_type == 'screen':
                db.collection('proctoring_sessions').document(session_id).update({
                    'screen_path': web_path,
                    'screen_uploaded_at': datetime.utcnow().isoformat()
                })
            else:
                db.collection('proctoring_sessions').document(session_id).update({
                    'video_path': web_path,
                    'video_uploaded_at': datetime.utcnow().isoformat()
                })
        except Exception as e:
            print('Warning: failed to update proctoring_sessions with video_path:', e)

        return jsonify({'message': 'uploaded', 'path': web_path}), 200
    except Exception as e:
        print('upload_video error', e)
        return jsonify({'error': str(e)}), 500


@app.route('/share_report', methods=['POST'])
def share_report():
    """
    Creates a shareable link for a specific report.
    Expects JSON with: student_id, exam_id, session_id
    Returns a unique token that can be used to access the report.
    """
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        exam_id = data.get('exam_id')
        session_id = data.get('session_id')

        for fid, fname in [(student_id, 'student_id'), (exam_id, 'exam_id'), (session_id, 'session_id')]:
            valid, err = validate_firestore_id(fid or '', fname)
            if not valid:
                return jsonify({'error': err}), 400

        # Generate unique token
        import secrets
        token = secrets.token_urlsafe(32)

        # Store share data
        share_data = {
            'token': token,
            'student_id': student_id,
            'exam_id': exam_id,
            'session_id': session_id,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': None  # No expiration for now
        }

        db.collection('shared_reports').add(share_data)

        # Generate shareable URL
        share_url = f"{request.host_url}shared_report/{token}"

        return jsonify({
            'success': True,
            'share_url': share_url,
            'token': token
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/shared_report/<token>')
def shared_report(token):
    """
    Displays a shared report using the token.
    No authentication required - private access via token.
    """
    try:
        # Find the shared report by token
        shares_ref = db.collection('shared_reports')
        query = shares_ref.where('token', '==', token).limit(1)
        share_doc = next(query.stream(), None)

        if not share_doc:
            return render_template('error.html', message='Invalid or expired share link'), 404

        share_data = share_doc.to_dict()

        # Check if expired (if we implement expiration later)
        if share_data.get('expires_at'):
            expires = datetime.fromisoformat(share_data['expires_at'])
            if datetime.utcnow() > expires:
                return render_template('error.html', message='Share link has expired'), 410

        # Get the report data
        student_id = share_data['student_id']
        exam_id = share_data['exam_id']
        session_id = share_data['session_id']

        # Fetch required data (similar to reports page logic)
        student_doc = db.collection('users').document(student_id).get()
        exam_doc = db.collection('exams').document(exam_id).get()
        session_doc = db.collection('proctoring_sessions').document(session_id).get()

        if not all([student_doc.exists, exam_doc.exists, session_doc.exists]):
            return render_template('error.html', message='Report data not found'), 404

        student = student_doc.to_dict()
        student['id'] = student_doc.id

        exam = exam_doc.to_dict()
        exam['id'] = exam_doc.id

        session = session_doc.to_dict()
        session['id'] = session_doc.id

        return render_template(
            'shared_report.html',
            student=student,
            exam=exam,
            session=session
        )

    except Exception as e:
        return render_template('error.html', message='Error loading report'), 500


@app.route('/start_session', methods=['POST'])
def start_session():
    """
    Start a new proctoring session.
    Expects JSON with: student_id, exam_id
    Returns: session_id
    """
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        exam_id = data.get('exam_id')

        valid, err = validate_firestore_id(student_id or '', 'student_id')
        if not valid:
            return jsonify({'error': err}), 400
        valid, err = validate_firestore_id(exam_id or '', 'exam_id')
        if not valid:
            return jsonify({'error': err}), 400

        session_data = {
            'student_id': student_id,
            'exam_id': exam_id,
            'start_time': datetime.utcnow().isoformat(),
            'status': 'active',
            'flags': {},
            'video_path': None,
            'clip_path': None,
            'screen_path': None
        }

        doc_ref = db.collection('proctoring_sessions').add(session_data)
        session_id = doc_ref[1].id

        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session started successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/end_session', methods=['POST'])
def end_session():
    """
    End an active proctoring session.
    Expects JSON with: session_id
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')

        valid, err = validate_session_id(session_id or '')
        if not valid:
            return jsonify({'error': err}), 400

        db.collection('proctoring_sessions').document(session_id).update({
            'end_time': datetime.utcnow().isoformat(),
            'status': 'completed'
        })

        return jsonify({
            'success': True,
            'message': 'Session ended successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_session_status')
def get_session_status():
    """
    Get the status of a proctoring session.
    Query param: session_id
    """
    try:
        session_id = request.args.get('session_id')
        valid, err = validate_session_id(session_id or '')
        if not valid:
            return jsonify({'error': err}), 400

        doc = db.collection('proctoring_sessions').document(session_id).get()
        if not doc.exists:
            return jsonify({'error': 'Session not found'}), 404

        session_data = doc.to_dict()
        session_data['id'] = doc.id

        return jsonify(session_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_or_create_session', methods=['POST'])
def get_or_create_session():
    """
    Get an existing active/paused proctoring session for a given student+exam,
    or create a new one if none exists.

    This is used so that violation flags (proctoring_sessions.flags) are
    preserved even if the student closes the browser and reopens the exam.

    Expects JSON:
    {
        "student_id": "...",
        "exam_id": "..."
    }

    Returns JSON:
    {
        "session_id": "...",
        "created": true|false,
        "session": { ...session_doc... }
    }
    """
    try:
        data = request.get_json() or {}
        student_id = data.get('student_id')
        exam_id = data.get('exam_id')

        valid, err = validate_firestore_id(student_id or '', 'student_id')
        if not valid:
            return jsonify({'error': err}), 400
        valid, err = validate_firestore_id(exam_id or '', 'exam_id')
        if not valid:
            return jsonify({'error': err}), 400

        sessions_ref = db.collection('proctoring_sessions')

        # Look for an existing session for this student+exam that is not completed/terminated
        query = (sessions_ref
                 .where('student_id', '==', student_id)
                 .where('exam_id', '==', exam_id)
                 .where('status', 'in', ['active', 'paused']))

        existing = list(query.stream())

        if existing:
            # Reuse the first matching session
            doc = existing[0]
            session_data = doc.to_dict()
            session_data['id'] = doc.id
            return jsonify({
                'created': False,
                'session_id': doc.id,
                'session': session_data
            }), 200

        # No existing active/paused session → create a new one
        session_data = {
            'student_id': student_id,
            'exam_id': exam_id,
            'start_time': datetime.utcnow().isoformat(),
            'status': 'active',
            # Keep flags empty so we only accumulate violations from this point
            'flags': {},
            'video_path': None,
            'clip_path': None,
            'screen_path': None
        }

        doc_ref = sessions_ref.add(session_data)
        session_id = doc_ref[1].id

        session_data['id'] = session_id

        return jsonify({
            'created': True,
            'session_id': session_id,
            'session': session_data
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def cleanup_old_reports():
    """
    Scheduled task to delete proctoring sessions and associated files older than 14 days.
    Also cleans up expired shared reports.
    """
    try:
        print("Starting scheduled cleanup of old reports...")

        # Calculate cutoff date (14 days ago)
        cutoff_date = datetime.utcnow() - timedelta(days=14)
        cutoff_iso = cutoff_date.isoformat()

        # Delete old proctoring sessions
        sessions_ref = db.collection('proctoring_sessions')
        old_sessions = sessions_ref.where('start_time', '<', cutoff_iso).stream()

        deleted_count = 0
        for session_doc in old_sessions:
            session_data = session_doc.to_dict()

            # Delete associated video files (webcam, clips, and screen recordings)
            for path_key in ['video_path', 'clip_path', 'screen_path']:
                if session_data.get(path_key):
                    # Convert web path to file system path (support both /recordings/ and legacy /static/recordings/)
                    web_path = session_data[path_key]
                    if web_path.startswith(WEB_RECORDINGS_PREFIX):
                        filename = web_path.replace(WEB_RECORDINGS_PREFIX, '')
                        file_path = os.path.join(RECORDINGS_DIR, filename)
                    elif web_path.startswith('/static/recordings/'):
                        filename = web_path.replace('/static/recordings/', '')
                        file_path = os.path.join(os.path.dirname(__file__), 'static', 'recordings', filename)
                    else:
                        file_path = None
                    if file_path:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {e}")

            # Delete the session document
            session_doc.reference.delete()
            deleted_count += 1

        print(f"Deleted {deleted_count} old proctoring sessions")

        # Clean up expired shared reports (if any have expiration set)
        shared_ref = db.collection('shared_reports')
        expired_shares = shared_ref.where('expires_at', '<', datetime.utcnow().isoformat()).stream()

        expired_count = 0
        for share_doc in expired_shares:
            share_doc.reference.delete()
            expired_count += 1

        print(f"Deleted {expired_count} expired shared reports")

    except Exception as e:
        print(f"Error during cleanup: {e}")


# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=cleanup_old_reports,
    trigger=CronTrigger(hour=2, minute=0),  # Run daily at 2 AM
    id='cleanup_old_reports',
    name='Clean up old proctoring reports',
    replace_existing=True
)
scheduler.start()

# Ensure scheduler shuts down gracefully
import atexit
atexit.register(lambda: scheduler.shutdown())


if __name__ == '__main__':
    # Check if SSL certificates exist
    ssl_cert = 'cert.pem'
    ssl_key = 'key.pem'

    if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        print("SSL certificates found. Starting HTTPS server...")
        app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=(ssl_cert, ssl_key))
    else:
        print("SSL certificates not found. Starting HTTP server...")
        print("Note: Camera access requires HTTPS. Generate SSL certificates for production use.")
        app.run(debug=True, host='0.0.0.0', port=5000)
