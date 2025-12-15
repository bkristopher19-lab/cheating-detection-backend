# Proctoring Cheating Detection AI

A Flask-based web application for real-time proctoring and cheating detection using computer vision and audio analysis.

## Features

- **Face Detection**: Detects multiple faces in the frame
- **Head Pose Estimation**: Tracks if the user is looking at the screen
- **Object Detection**: Uses YOLO to detect prohibited objects (phones, laptops, books, etc.)
- **Audio Analysis**: Detects loud sounds and suspicious audio patterns
- **Frontend Alerts**: Captures browser-based cheating behaviors (tab switches, keyboard shortcuts, etc.)
- **Session Management**: Tracks proctoring sessions with Firebase Firestore
- **Video Recording**: Supports video upload and storage for review

## Prerequisites

- Python 3.11+
- Firebase Admin SDK credentials (`firebase_admin_key.json`)
- YOLO model file (optional, for object detection)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Firebase:
   - Place your Firebase Admin SDK credentials file as `firebase_admin_key.json` in the root directory
   - Ensure your Firestore database is properly configured

5. (Optional) Set up YOLO model:
   - Place `yolov5s.onnx` in a `models/` directory for object detection
   - Or use the existing `yolov9-m.pt` file (converted to ONNX if needed)

## Running the Application

### Development Mode
```bash
python detection.py
```

The application will run on `http://localhost:5000`

### Production Mode
Using Gunicorn (as specified in Procfile):
```bash
gunicorn detection:app
```

### HTTPS Setup
For camera access in browsers, HTTPS is required. Place SSL certificates as:
- `cert.pem` (SSL certificate)
- `key.pem` (SSL private key)

The application will automatically use HTTPS if these files are present.

## API Endpoints

### Detection Endpoints
- `POST /predict` - Process video frame for cheating detection
- `POST /detect_audio` - Analyze audio for suspicious sounds
- `POST /frontend_alert` - Receive browser-based cheating alerts

### Session Management
- `POST /start_session` - Start a new proctoring session
- `POST /end_session` - End an active session
- `GET /get_session_status` - Get session status

### Video Management
- `POST /upload_video` - Upload video recording for a session

### Data Management
- `POST /add` - Add document to Firestore collection
- `GET /select` - Query Firestore collection
- `POST /edit/<id>` - Update document
- `POST /delete/<id>` - Delete document

### Authentication
- `POST /login` - User authentication
- `POST /logout` - User logout

## Project Structure

```
.
├── detection.py              # Main Flask application
├── requirements.txt          # Python dependencies
├── templates/                # HTML templates
├── static/                   # Static files (images, recordings, sounds)
├── firebase_admin_key.json   # Firebase Admin SDK credentials
├── Procfile                  # Deployment configuration
└── ProctoringApp.spec        # PyInstaller spec file (for packaging)
```

## Technologies Used

- **Flask**: Web framework
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Face and hand landmark detection
- **YOLO**: Object detection
- **Firebase Firestore**: Database and session storage
- **Librosa/Pydub**: Audio processing
- **APScheduler**: Scheduled tasks for cleanup

## Deployment

The application can be deployed to platforms like Render, Heroku, or any Python-compatible hosting service. The `Procfile` is configured for Gunicorn deployment.

## License

[Add your license here]
