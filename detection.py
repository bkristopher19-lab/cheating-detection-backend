import os
import cv2
import numpy as np
import mediapipe as mp
import base64
import math
from werkzeug.utils import secure_filename
from datetime import datetime

# Web server
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for

# Firebase
from firebase_admin import credentials, firestore
import firebase_admin

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection


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
            cv2.circle(image, (x, y), 1, (80,110,10), -1)

    # Draw left hand landmarks and connections
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

    # Draw right hand landmarks and connections
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )


def extract_keypoints(results):
    # keep helper if later needed; returns concatenated arrays (pose excluded to focus on face+hands)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, lh, rh])


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Initialize Firebase
cred = credentials.Certificate(r"firebase_admin_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def count_faces_bounding_boxes(image, min_confidence=0.4):
    """Return number of faces detected and list of bounding boxes (x,y,w,h) in pixel coords."""
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_confidence) as fd:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = fd.process(img_rgb)
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
if os.path.exists(YOLO_ONNX):
    try:
        yolo_net = cv2.dnn.readNet(YOLO_ONNX)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        yolo_net = None
else:
    yolo_net = None


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

        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
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

@app.route('/predict', methods=['POST'])
def predict():
    session_id = request.args.get('session_id')

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
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # estimate head pose / gaze
        looking, angles = estimate_head_pose_from_mesh(results, image)



    # overlay hints and boxes
    if face_count:
        for (x,y,wc,hc) in face_boxes:
            cv2.rectangle(image, (x,y), (x+wc, y+hc), (0,200,0), 2)
    cv2.putText(image, f"Faces: {face_count}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2, cv2.LINE_AA)


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
            if face_count == 0:
                flags['face_absent'] = [{'timestamp': datetime.utcnow().isoformat(), 'message': 'No face detected'}]
            if face_count > 1:
                flags['multiple_faces'] = [{'timestamp': datetime.utcnow().isoformat(), 'message': f'Multiple faces detected ({face_count})'}]
            if not looking:
                flags['not_looking'] = [{'timestamp': datetime.utcnow().isoformat(), 'message': 'Student not looking at screen'}]

            if flags:
                # Update proctoring_sessions with flags
                db.collection('proctoring_sessions').document(session_id).update({
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
        'head_angles': {'yaw': angles[0], 'pitch': angles[1], 'roll': angles[2]} if angles else None,
    })


# Update the existing CRUD routes to handle different collections

@app.route('/add', methods=['POST'])
def add():
    try:
        collection = request.args.get('collection', 'users')
        data = request.json
        doc_ref = db.collection(collection).add(data)
        return jsonify({'id': doc_ref[1].id}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select')
def select():
    try:
        collection = request.args.get('collection', 'users')
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
        db.collection(collection).document(id).update(request.json)
        return jsonify({'message': 'Updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<id>', methods=['POST'])
def delete(id):
    try:
        collection = request.args.get('collection', 'users')
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

@app.route('/register')
def register_page():
    return render_template('register.html')

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

@app.route('/reports')
def reports():
    return render_template('reports.html')
 
@app.route('/take_exam/<exam_id>')
def take_exam(exam_id):
    exam_ref = db.collection('exams').document(exam_id)
    exam = exam_ref.get()
    
    if not exam.exists:
        return redirect('/student_dashboard')
        
    return render_template('index.html', exam_id=exam_id)

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/proctor_dashboard')
def proctor_dashboard():
    return render_template('proctor_dashboard.html')

@app.route('/logout', methods=['POST'])
def logout():
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Accepts multipart/form-data video upload and attaches it to a proctoring session.
    Query param: session_id (required) or form field session_id
    Query param: type (optional, 'clip' for cheating clips)
    Saves file under /static/recordings/ and updates proctoring_sessions document with video_path or clip_path.
    """
    try:
        session_id = request.args.get('session_id') or request.form.get('session_id')
        upload_type = request.args.get('type', 'full')  # 'full' or 'clip'
        if not session_id:
            return jsonify({'error': 'session_id required'}), 400

        if 'video' not in request.files:
            return jsonify({'error': 'no video file sent'}), 400

        f = request.files['video']
        if f.filename == '':
            return jsonify({'error': 'empty filename'}), 400

        safe_name = secure_filename(f.filename)
        recordings_dir = os.path.join(os.path.dirname(__file__), 'static', 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        filename = f"session_{session_id}_{timestamp}_{safe_name}"
        save_path = os.path.join(recordings_dir, filename)
        f.save(save_path)

        web_path = f"/static/recordings/{filename}"

        # update proctoring_sessions document
        try:
            if upload_type == 'clip':
                db.collection('proctoring_sessions').document(session_id).update({
                    'clip_path': web_path,
                    'clip_uploaded_at': datetime.utcnow().isoformat()
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

        if not all([student_id, exam_id, session_id]):
            return jsonify({'error': 'student_id, exam_id, and session_id are required'}), 400

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

        return render_template('shared_report.html',
                             student=student,
                             exam=exam,
                             session=session)

    except Exception as e:
        return render_template('error.html', message='Error loading report'), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_admin_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()
