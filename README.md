# Proctoring Cheating Detection AI

A Flask-based web application for real-time proctoring and cheating detection using computer vision and audio analysis. Optimized for Render free tier hosting with memory-efficient processing.

## If you want to debug in flask in line 160+ detection py
# Initialize Firebase
if not firebase_admin._apps:
    # Use environment variable for key path, default to local file for development
    key_path = os.getenv('FIREBASE_KEY_PATH', 'firebase_admin_key.json')
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

## If you want to finally commit and push to git
if not firebase_admin._apps:
    cred = credentials.Certificate("/etc/secrets/firebase_admin_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()


git ls-files | findstr firebase_admin_key.json
git ls-files | findstr adminsdk

git add .
git commit -m "Update backend for Render"
git pull --rebase origin main
git push origin main

## Features

### Detection Capabilities
- **Face Detection**: Detects multiple faces in the frame (MediaPipe)
- **Head Pose Estimation**: Tracks if the user is looking at the screen
- **Object Detection**: Uses YOLO to detect prohibited objects (phones, laptops, books, etc.)
- **Audio Analysis**: Detects loud sounds and suspicious audio patterns (configurable threshold)
- **Frontend Alerts**: Captures browser-based cheating behaviors:
  - Tab switches (with 3-second grace period)
  - Keyboard shortcuts (Ctrl+C, Ctrl+V, etc.)
  - Right-click attempts
  - Fullscreen exit
  - Window resize
  - Copy/paste attempts

### Exam Security
- **Full-Screen Mode**: Required throughout exam duration
- **Violation System**: 
  - Most violations: 3 strikes (warning → pause → termination)
  - Phone detection: Immediate termination (1 strike)
  - 5-second pause after each violation (except 3rd)
  - Automatic exam termination with 0 score on 3rd violation
- **Session Recording**: Full exam session video recording with cheating clip capture

### User Management
- **Role-Based Access**: Admin, Proctor, and Student roles
- **Section/Class Management**: Organize students into classes with sections
- **Join Code System**: Students can join sections using unique join codes
- **Exam Filtering**: Students only see exams assigned to their section
- **User Assignment**: Admins/Proctors can assign students to classes via UI

### Dashboard & Analytics
- **Student Dashboard**: View available exams, scores, and section information
- **Proctor Dashboard**: Monitor student exam participation, scores, and proctoring flags
- **Admin Dashboard**: Full system analytics and management
- **Reports**: Detailed proctoring reports with flags and video recordings

## Prerequisites

- Python 3.11+
- Firebase Admin SDK credentials (`firebase_admin_key.json`)
- YOLO model file (optional, for object detection)
- Modern web browser with camera/microphone access

## Installation

1. **Create a virtual environment:**
```bash
python -m venv venv
```

2. **Activate the virtual environment:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Firebase:**
   - Place your Firebase Admin SDK credentials file as `firebase_admin_key.json` in the root directory
   - Ensure your Firestore database is properly configured
   - Create collections: `users`, `exams`, `classes`, `proctoring_sessions`, `exam_results`, `shared_reports`

5. **(Optional) Set up YOLO model:**
   - Place `yolov5s.onnx` in a `models/` directory for object detection
   - Or use the existing `yolov9-m.pt` file (convert to ONNX if needed)

## Configuration

### Audio Detection Threshold
Edit `detection.py` line 33:
```python
AUDIO_LOUD_THRESHOLD = 0.60  # Adjust sensitivity (0.0 to 1.0)
# 0.60 = detects background noise and normal talking
# 0.90 = only very loud sounds (shouting)
```

### Server-Side Processing Frequency
Edit `templates/index.html` line ~1657:
```javascript
let intervalMs = 15000; // Default: 15 seconds (PC and mobile similar light load)
// Set to 0 to disable server-side processing (client-side only)
// For paid hosting: Can use 500-1000ms for real-time detection
```

### Frame Size (Memory Optimization)
Default: 640×480 pixels (optimized for Render free tier)
- Can be adjusted in `templates/index.html` getUserMedia constraints
- Larger sizes increase memory usage

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
- `POST /predict` - Process video frame for cheating detection (optimized for memory)
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

### Authentication & User Management
- `POST /login` - User authentication
- `POST /logout` - User logout
- `GET /student_dashboard` - Student dashboard
- `GET /proctor_dashboard` - Proctor dashboard
- `GET /dashboard` - Admin dashboard

## Project Structure

```
.
├── detection.py              # Main Flask application
├── requirements.txt          # Python dependencies
├── templates/                # HTML templates
│   ├── index.html           # Exam interface (main proctoring page)
│   ├── student_dashboard.html
│   ├── proctor_dashboard.html
│   ├── dashboard.html       # Admin dashboard
│   ├── users.html           # User management
│   ├── classes.html         # Class/section management
│   ├── exams.html           # Exam creation/management
│   └── ...
├── static/                   # Static files
│   ├── images/              # Images and logos
│   ├── recordings/          # Video recordings (ephemeral on Render)
│   └── sounds/              # Alert sounds
├── models/                   # ML models directory (optional)
│   └── yolov5s.onnx         # YOLO model (if used)
├── firebase_admin_key.json   # Firebase Admin SDK credentials
├── Procfile                  # Deployment configuration
└── ProctoringApp.spec        # PyInstaller spec file (for packaging)
```

## Technologies Used

- **Flask**: Web framework
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Face and hand landmark detection (client-side and server-side)
- **YOLO**: Object detection (optional)
- **Firebase Firestore**: Database and session storage
- **Librosa/Pydub**: Audio processing
- **APScheduler**: Scheduled tasks for cleanup
- **COCO-SSD**: Client-side object detection (browser)

## Memory Optimization (Render Free Tier)

The application is optimized to avoid Exit status 137 (memory limit exceeded):

1. **ML Models Loaded Once**: MediaPipe and YOLO models are loaded at startup, not per request
2. **Reduced Frame Size**: 640×480 pixels (67% less memory than 1280×720)
3. **Low Processing Frequency**: 5-second intervals (0.2 FPS to server)
4. **Frame Resizing**: YOLO processes resized frames (640×640 max)
5. **Client-Side Processing**: Most detection runs in browser (MediaPipe Face Mesh, COCO-SSD)

### Expected Performance
- **1-5 students**: Works smoothly
- **5-10 students**: May experience occasional delays
- **10+ students**: Consider upgrading to paid tier or disabling server-side processing

### Disable Server-Side Processing
To maximize efficiency, set `intervalMs = 0` in `templates/index.html`:
- Client-side detection still works (face detection, phone detection, tab switches, etc.)
- Server only handles data storage and session management
- Reduces server load by ~90%

## Deployment

### Render Free Tier
1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn detection:app`
4. Add environment variables if needed
5. **Important**: Video recordings in `static/recordings/` are ephemeral (lost on restart)
   - Consider using Firebase Storage or AWS S3 for persistent storage

### Other Platforms
The application can be deployed to:
- **Heroku**: Use Procfile
- **AWS Elastic Beanstalk**: Python platform
- **DigitalOcean App Platform**: Python app
- **Any Python-compatible hosting service**

## Usage Guide

### For Administrators
1. Create classes/sections with unique join codes
2. Assign proctors to classes
3. Create exams and assign them to specific sections
4. Monitor system-wide analytics

### For Proctors
1. View exams assigned to your sections
2. Monitor student exam participation
3. Review proctoring flags and scores
4. Filter by exam and section

### For Students
1. Join a section using the join code provided by your instructor
2. View available exams for your section
3. Take exams with full-screen proctoring
4. View your exam scores and results

## Violation Types & Limits

| Violation Type | Max Strikes | Action on Limit |
|---------------|-------------|-----------------|
| Phone Detection | 1 | Immediate termination (0 score) |
| Multiple Faces | 3 | Warning → Pause → Termination |
| Loud Sound | 3 | Warning → Pause → Termination |
| Fullscreen Exit | 3 | Warning → Pause → Termination |
| Tab Switch | 3 | Warning → Pause → Termination |
| Keyboard Shortcut | 3 | Warning → Pause → Termination |
| Right Click | 3 | Warning → Pause → Termination |
| Face Absent | 3 | Warning → Pause → Termination |

## Troubleshooting

### Camera/Microphone Not Working
- Ensure HTTPS is enabled (required for camera access)
- Check browser permissions
- Verify SSL certificates are present

### Memory Issues on Render
- Increase `intervalMs` to 10000 (10 seconds)
- Set `intervalMs = 0` to disable server-side processing
- Check Render logs for memory usage

### Exams Not Showing for Students
- Verify student is assigned to a class (`class_id` in user document)
- Verify exam is assigned to student's class (`class_id` in exam document)
- Check exam start/end dates

### Models Not Loading
- Ensure `models/` directory exists
- Verify YOLO model file is in ONNX format
- Check file permissions

## License

[Add your license here]

## Support

For issues or questions, please check:
- Firebase Firestore console for data issues
- Browser console for client-side errors
- Server logs for backend errors
- Render logs for deployment issues
