# Detection.py Comprehensive Analysis

## Overview
`detection.py` is the core Flask application that powers the proctoring cheating detection system. It handles real-time video/audio analysis, session management, user authentication, and data persistence via Firebase Firestore.

---

## 🎓 Beginner's Guide: How This Code Works

### What is Flask? (Think of it like a Restaurant Waiter)
Imagine Flask as a waiter in a restaurant:
- **The Customer (Browser)**: Makes a request like "I want to see the login page" or "Process this video frame"
- **The Waiter (Flask)**: Takes the order, goes to the kitchen (your code), brings back the result
- **The Kitchen (Your Functions)**: Does the actual work (detects faces, saves data, etc.)
- **The Menu (Routes)**: Different URLs like `/login`, `/predict` are like menu items

**In Code Terms:**
```python
@app.route('/predict')  # This is like a menu item
def predict():          # This is the "recipe" (function)
    # Do something here
    return result       # Return the "dish" (response)
```

When someone visits `http://yoursite.com/predict`, Flask runs the `predict()` function and returns the result.

### How HTTP Requests Work (Like Sending a Letter)
1. **Request**: Browser sends a "letter" (HTTP request) to the server
   - Contains: What you want (`GET` = "give me data", `POST` = "here's data to process")
   - Contains: The address (`/predict`, `/login`, etc.)
   - Contains: Data (like a video frame or login credentials)

2. **Processing**: Server receives the letter, reads it, does the work

3. **Response**: Server sends back a "reply letter" (HTTP response)
   - Contains: Status code (200 = "OK", 404 = "Not Found", 500 = "Error")
   - Contains: Data (like JSON with detection results or an HTML page)

**Example Flow:**
```
Browser: "Hey server, POST this video frame to /predict"
Server: "Got it! Processing... Here's what I found: {face_count: 1, looking: true}"
Browser: "Thanks! I'll show this to the user"
```

### How the Code is Structured
Think of `detection.py` like a building with different floors:

1. **Ground Floor (Lines 1-38)**: Setup and Imports
   - Like bringing in tools and materials
   - Imports libraries (Flask, OpenCV, MediaPipe, etc.)
   - Sets up configuration (like audio threshold)

2. **First Floor (Lines 40-284)**: Helper Functions
   - Like specialized workers who do specific tasks
   - `mediapipe_detection()` - processes images
   - `count_faces_bounding_boxes()` - counts faces
   - `estimate_head_pose_from_mesh()` - checks if looking at screen

3. **Second Floor (Lines 86-164)**: Flask App Setup
   - Creates the Flask app (the "waiter")
   - Connects to Firebase (the database)
   - Loads AI models (like MediaPipe) once at startup

4. **Third Floor (Lines 287-1238)**: API Endpoints (Routes)
   - Each `@app.route()` is like a different service
   - `/predict` - processes video frames
   - `/detect_audio` - analyzes audio
   - `/login` - handles user login
   - etc.

### How Data Flows Through the System

**Example: When a Student Takes an Exam**

```
1. Student clicks "Start Exam" button
   ↓
2. Browser (JavaScript) sends request: POST /add?collection=proctoring_sessions
   ↓
3. Flask receives request → Runs the /add function
   ↓
4. Function creates document in Firebase Firestore (database)
   ↓
5. Firebase returns: {"id": "session_12345"}
   ↓
6. Flask sends back to browser: {"id": "session_12345"}
   ↓
7. Browser stores session_id and starts camera
   ↓
8. Every 15 seconds, browser sends video frame: POST /predict?session_id=12345
   ↓
9. Flask processes frame:
   - Counts faces
   - Checks head pose
   - Detects objects (phones, etc.)
   ↓
10. Flask saves flags to Firebase: proctoring_sessions/session_12345/flags
   ↓
11. Flask sends back results: {"face_count": 1, "looking_at_screen": true}
   ↓
12. Browser displays results and shows alerts if violations detected
```

### How Different Parts Connect Together

**The Big Picture:**
```
┌─────────────┐
│   Browser   │  (Frontend - templates/index.html)
│  (Student)  │
└──────┬──────┘
       │ HTTP Requests (POST/GET)
       ↓
┌─────────────────────────────────┐
│      Flask Server                │
│      (detection.py)              │
│                                  │
│  ┌──────────────────────────┐   │
│  │  API Endpoints            │   │
│  │  /predict, /login, etc.   │   │
│  └───────────┬──────────────┘   │
│              │                    │
│  ┌───────────▼──────────────┐   │
│  │  Helper Functions         │   │
│  │  (Face detection, etc.)    │   │
│  └───────────┬──────────────┘   │
│              │                    │
│  ┌───────────▼──────────────┐   │
│  │  AI Models                │   │
│  │  (MediaPipe, YOLO)        │   │
│  └──────────────────────────┘   │
└──────────┬───────────────────────┘
           │
           │ Database Operations
           ↓
┌──────────────────────────┐
│   Firebase Firestore      │
│   (Cloud Database)        │
│                          │
│   - users                │
│   - exams                │
│   - proctoring_sessions  │
│   - exam_results         │
└──────────────────────────┘
```

**In Simple Terms:**
- **Browser** = The user interface (what students see)
- **Flask** = The brain (processes everything)
- **Firebase** = The memory (stores all data)
- **AI Models** = The eyes (detect faces, objects, etc.)

---

## Architecture & Dependencies

### Core Technologies
- **Flask 3.0.3**: Web framework and routing
- **OpenCV (cv2)**: Computer vision processing, image manipulation
- **MediaPipe**: Face detection, holistic pose estimation (face + hands)
- **Firebase Admin SDK**: Firestore database operations
- **Librosa/Pydub**: Audio processing and analysis
- **APScheduler**: Background scheduled tasks for cleanup
- **NumPy**: Numerical operations for image/audio processing

### Key Imports
```python
- Flask, render_template, Response, request, jsonify, redirect, url_for, flash
- firebase_admin (credentials, firestore)
- cv2 (OpenCV)
- mediapipe (holistic, face_detection, drawing_utils)
- librosa, pydub (AudioSegment)
- apscheduler (BackgroundScheduler, CronTrigger)
```

---

## Core Detection Functions

### 1. MediaPipe Detection Pipeline

#### `mediapipe_detection(image, model)` (Lines 40-46)
- **Purpose**: Processes image through MediaPipe holistic model
- **Process**: 
  - Converts BGR → RGB
  - Processes with MediaPipe model
  - Converts back to BGR
- **Returns**: Processed image, MediaPipe results object

**🔍 How It Works (Simple Explanation):**
Think of this function like a photo scanner at a security checkpoint:
1. **Input**: A photo (image) comes in
2. **Color Conversion**: The image is in BGR format (Blue-Green-Red, OpenCV's default), but MediaPipe needs RGB (Red-Green-Blue, like normal photos). So we flip the colors around.
3. **Processing**: MediaPipe AI model analyzes the image - like a security guard looking for faces, hands, body posture
4. **Color Conversion Back**: Convert back to BGR so OpenCV can work with it
5. **Output**: The same image, plus a "report" (results object) saying what was found

**In Code:**
```python
def mediapipe_detection(image, model):
    # Step 1: Change color format (BGR → RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 2: Tell MediaPipe "don't modify the image, just analyze it"
    image.flags.writeable = False
    
    # Step 3: Run the AI model - this is where the magic happens!
    results = model.process(image)
    
    # Step 4: Now we can modify the image again
    image.flags.writeable = True
    
    # Step 5: Change back to BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Return both: the image and what was detected
    return image, results
```

#### `draw_styled_landmarks(image, results)` (Lines 49-75)
- **Purpose**: Visualizes MediaPipe detections on image
- **Features**:
  - Face landmarks as small filled dots (green)
  - Left hand landmarks (purple/pink)
  - Right hand landmarks (orange/pink)
- **Used for**: Visual feedback and debugging

**🎨 How It Works (Simple Explanation):**
This function is like drawing dots and lines on a photo to show what the AI detected:
- **Face**: Draws 468 tiny green dots showing face features (eyes, nose, mouth, etc.)
- **Left Hand**: Draws 21 purple/pink dots connected by lines showing finger positions
- **Right Hand**: Draws 21 orange/pink dots connected by lines

**Why?** So developers can see what the AI is detecting. Like highlighting important parts of a photo.

**In Code:**
```python
def draw_styled_landmarks(image, results):
    # If a face was detected...
    if results.face_landmarks:
        # Get image size
        h, w, _ = image.shape
        
        # Draw a tiny dot for each face landmark (468 dots total)
        for landmark in results.face_landmarks.landmark:
            x = int(landmark.x * w)  # Convert percentage to pixel position
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 1, (80, 110, 10), -1)  # Draw green dot
    
    # Similar for left and right hands...
```

#### `extract_keypoints(results)` (Lines 78-83)
- **Purpose**: Extracts numerical keypoints for ML model integration
- **Returns**: Concatenated numpy array of:
  - Face landmarks: 468 points × 3 (x, y, z) = 1404 values
  - Left hand: 21 points × 3 = 63 values
  - Right hand: 21 points × 3 = 63 values
  - **Total**: 1530 values per frame
- **Note**: Pose landmarks excluded to focus on face/hands

**📊 How It Works (Simple Explanation):**
This function converts the detected landmarks into a list of numbers. Think of it like:
- **Face**: 468 points, each with 3 coordinates (x, y, z) = 1,404 numbers
- **Left Hand**: 21 points × 3 = 63 numbers
- **Right Hand**: 21 points × 3 = 63 numbers
- **Total**: 1,530 numbers describing the person's face and hands

**Why?** Machine learning models need numbers, not visual dots. This converts the visual detection into a format that could be used for training AI models or storing patterns.

**In Code:**
```python
def extract_keypoints(results):
    # Extract face landmarks as numbers: [x1, y1, z1, x2, y2, z2, ...]
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    # If no face detected, use zeros instead
    if not results.face_landmarks:
        face = np.zeros(468 * 3)  # 1404 zeros
    
    # Same for hands...
    left_hand = ...  # 63 numbers
    right_hand = ...  # 63 numbers
    
    # Combine all into one big list: [face_numbers..., left_hand..., right_hand...]
    return np.concatenate([face, left_hand, right_hand])
```

### 2. Face Detection

#### `count_faces_bounding_boxes(image, min_confidence=0.4)` (Lines 167-184)
- **Purpose**: Counts faces and returns bounding boxes
- **Model**: MediaPipe FaceDetection (loaded once at startup)
- **Returns**: 
  - Face count (integer)
  - List of bounding boxes [(x, y, width, height)]
- **Confidence threshold**: 0.4 (configurable)
- **Global model**: `face_detection_model` (reused across requests)

**👤 How It Works (Simple Explanation):**
This function is like a bouncer counting people at a club:
1. **Input**: A photo
2. **AI Analysis**: MediaPipe FaceDetection model scans the image
3. **Confidence Check**: Only counts faces if the AI is at least 40% sure it's a face (min_confidence=0.4)
4. **Bounding Boxes**: For each face found, draws an imaginary rectangle around it
5. **Output**: 
   - How many faces (e.g., "2 faces detected")
   - Where each face is (rectangle coordinates: x, y, width, height)

**Why Bounding Boxes?** They tell you WHERE the face is in the image, like coordinates on a map.

**In Code:**
```python
def count_faces_bounding_boxes(image, min_confidence=0.4):
    # Convert image to RGB (MediaPipe needs RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run face detection AI model
    res = face_detection_model.process(img_rgb)
    
    # If no faces found, return 0
    if not res.detections:
        return 0, []
    
    # Get image dimensions
    h, w = image.shape[:2]
    boxes = []
    
    # For each face detected...
    for detection in res.detections:
        # Get the bounding box (rectangle coordinates)
        bbox = detection.location_data.relative_bounding_box
        
        # Convert from percentages (0.0-1.0) to pixel coordinates
        x = int(bbox.xmin * w)      # Left edge
        y = int(bbox.ymin * h)      # Top edge
        width = int(bbox.width * w)  # Width
        height = int(bbox.height * h) # Height
        
        boxes.append((x, y, width, height))
    
    return len(boxes), boxes  # Return count and list of boxes
```

**Example Output:**
- Face count: `2`
- Bounding boxes: `[(100, 50, 200, 250), (400, 60, 180, 240)]`
  - First face: at x=100, y=50, width=200, height=250
  - Second face: at x=400, y=60, width=180, height=240

### 3. Head Pose Estimation

#### `estimate_head_pose_from_mesh(results, image)` (Lines 216-284)
- **Purpose**: Determines if user is looking at screen
- **Method**: 6-point correspondence using solvePnP
- **Landmark indices**: [1, 199, 33, 263, 61, 291]
  - Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
- **Returns**: 
  - `(looking_bool, (yaw, pitch, roll))` or `(False, None)`
- **Thresholds**:
  - Looking: `abs(yaw) < 30°` AND `abs(pitch) < 20°`
  - Loosened thresholds for robustness
- **Uses**: OpenCV solvePnP with standard 3D model points

**🧭 How It Works (Simple Explanation):**
This function is like checking if someone is looking at you:
1. **6 Key Points**: Uses 6 specific points on the face (nose tip, chin, eye corners, mouth corners)
2. **3D Math**: Compares these points to a standard 3D face model to calculate head rotation
3. **Angles Calculated**:
   - **Yaw**: Left/right rotation (like shaking head "no")
   - **Pitch**: Up/down rotation (like nodding "yes")
   - **Roll**: Tilting left/right (like tilting head)
4. **Decision**: If yaw < 30° AND pitch < 20°, person is "looking at screen"

**Why 6 Points?** Like using landmarks to figure out where you are on a map. These 6 points are enough to determine head orientation.

**In Code (Simplified):**
```python
def estimate_head_pose_from_mesh(results, image):
    # Step 1: Get 6 specific face landmarks
    # Point 1: Nose tip
    # Point 2: Chin
    # Point 3: Left eye corner
    # Point 4: Right eye corner
    # Point 5: Left mouth corner
    # Point 6: Right mouth corner
    
    # Step 2: Convert to pixel coordinates
    image_points = [(landmark.x * width, landmark.y * height) for landmark in landmarks]
    
    # Step 3: Compare to standard 3D face model (like a template)
    # This uses complex math (solvePnP) to figure out rotation
    
    # Step 4: Calculate angles
    yaw = ...   # Left/right angle
    pitch = ... # Up/down angle
    roll = ...  # Tilt angle
    
    # Step 5: Check if looking at screen
    looking = (abs(yaw) < 30) and (abs(pitch) < 20)
    
    return looking, (yaw, pitch, roll)
```

**Example Output:**
- `(True, (5.2, -3.1, 0.8))` = Looking at screen, slight right turn, slight down tilt
- `(False, (45.0, 10.0, 0.0))` = Not looking (head turned 45° to the right)

---

## API Endpoints

### Detection Endpoints

#### `POST /predict` (Lines 382-619)
**Purpose**: Main video frame processing endpoint for cheating detection

**Input Methods** (supports multiple):
1. Multipart form-data: `frame`, `file`, or `image` field
2. Form field: base64 encoded image
3. JSON payload: `image`, `image_b64`, or `frame` field
4. Raw request data: base64 or data URI

**Query Parameters**:
- `session_id` (optional): Links detection to proctoring session

**Processing Pipeline**:
1. **Face Detection**: Counts faces using MediaPipe FaceDetection
2. **Holistic Processing**: MediaPipe holistic model (face mesh + hands)
3. **Head Pose**: Estimates gaze direction
4. **Keypoint Extraction**: Extracts 1530-dimensional feature vector
5. **Object Detection**: YOLO (if model available) for phones/books/laptops
   - Resizes to 640×640 for memory efficiency
   - Scales coordinates back to original frame size
   - Confidence threshold: 0.5

**Detection Flags**:
- Multiple faces (`twofaces`): When `face_count > 1`
- Object detection: Flags for `cell phone`, `laptop`, `book`, `remote`, `keyboard`
- Flags stored in Firestore `proctoring_sessions.flags`

**Response**:
```json
{
  "processed_image": "data:image/jpeg;base64,...",
  "face_count": 1,
  "looking_at_screen": true,
  "head_angles": {"yaw": 5.2, "pitch": -3.1, "roll": 0.8},
  "detected_objects": [...],
  "keypoints": [1530 values],
  "label": null,
  "probability": 0.0
}
```

**Memory Optimization**:
- Models loaded once at startup (not per request)
- Frame resizing for YOLO (640×640 max)
- Optimized for Render free tier (30-second timeout limit)

**🎬 How It Works (Step-by-Step for Beginners):**

**Step 1: Receiving the Image**
The browser sends a video frame (like a single photo from a video). The code tries multiple ways to get it:
- As a file upload (like uploading a photo)
- As base64 text (image converted to text format)
- As JSON data

**Step 2: Decoding the Image**
```python
# The image comes as bytes (like a compressed file)
# We need to convert it to something OpenCV can work with
nparr = np.frombuffer(data, np.uint8)  # Convert bytes to numbers
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode into image
```

**Step 3: Face Detection**
```python
# Count how many faces are in the image
face_count, face_boxes = count_faces_bounding_boxes(frame)
# If face_count > 1, someone else might be helping (cheating!)
```

**Step 4: Holistic Analysis**
```python
# Analyze face, hands, and body pose
image, results = mediapipe_detection(frame, holistic_model)
# This gives us detailed information about the person
```

**Step 5: Head Pose Check**
```python
# Check if person is looking at screen
looking, angles = estimate_head_pose_from_mesh(results, image)
# If not looking, they might be looking at notes or phone
```

**Step 6: Object Detection (YOLO)**
```python
# Check for prohibited objects (phones, books, etc.)
# First, resize image to save memory (640x640 instead of full size)
frame_resized = cv2.resize(frame, (640, 640))
# Run YOLO AI model
detected_objects = yolo_net.detect(frame_resized)
# Filter for cheating-related objects
```

**Step 7: Save Flags to Database**
```python
# If violations found, save them to Firebase
if face_count > 1:
    flags['twofaces'] = [{
        'timestamp': datetime.utcnow().isoformat(),
        'message': f'Multiple faces detected ({face_count})'
    }]
# Update database
db.collection('proctoring_sessions').document(session_id).update({'flags': flags})
```

**Step 8: Return Results**
```python
# Send back everything we found
return jsonify({
    'processed_image': data_uri,  # Image with drawings on it
    'face_count': face_count,
    'looking_at_screen': looking,
    'detected_objects': detected_objects,
    # ... etc
})
```

**Real-World Analogy:**
Think of `/predict` like a security checkpoint:
1. **Photo arrives** (video frame)
2. **Count people** (face detection)
3. **Check if looking at camera** (head pose)
4. **Scan for prohibited items** (object detection)
5. **Log violations** (save flags)
6. **Send report** (return results)

**Why Multiple Input Methods?**
Different browsers/devices send data differently. The code handles all formats so it works everywhere.

#### `POST /detect_audio` (Lines 287-379)
**Purpose**: Audio analysis for loud sound detection

**Input Methods**:
1. Multipart form-data: `audio` file
2. JSON: `audio` or `audio_b64` field
3. Form data: `audio` or `audio_b64` field

**Query Parameters**:
- `session_id` (optional): Links detection to session

**Processing**:
1. Decodes audio (WAV/PCM or raw PCM)
2. Converts to mono if stereo
3. Normalizes to float32 (0.0-1.0 range)
4. Calculates RMS (Root Mean Square) for loudness
5. Compares against `AUDIO_LOUD_THRESHOLD` (0.60)

**Configuration** (Line 33):
```python
AUDIO_LOUD_THRESHOLD = 0.60  # RMS threshold (0.0 to 1.0)
# 0.60 = detects background noise and normal talking
# 0.75 = moderate, detects normal talking
# 0.90+ = only very loud sounds (shouting)
```

**Response**:
```json
{
  "loud_sound": true,
  "rms_level": 0.65
}
```

**Flags**: Stores `loud_sound` flag in Firestore if threshold exceeded

**🔊 How It Works (Simple Explanation):**

**What is RMS?** Root Mean Square is a way to measure how loud sound is. Think of it like averaging the "energy" of the sound wave.

**Step-by-Step Process:**

1. **Receive Audio Data**
   ```python
   # Audio comes as bytes (like a music file)
   audio_data = request.files['audio'].read()
   ```

2. **Decode Audio**
   ```python
   # Convert audio bytes into numbers (samples)
   audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
   samples = np.array(audio_segment.get_array_of_samples())
   # Now we have a list of numbers representing sound waves
   ```

3. **Convert to Mono (Single Channel)**
   ```python
   # If stereo (2 channels), combine into one
   if audio_segment.channels == 2:
       samples = samples.mean(axis=1)  # Average left and right
   ```

4. **Normalize (Scale to 0.0-1.0)**
   ```python
   # Audio samples are usually -32768 to +32768
   # Convert to 0.0 to 1.0 range for easier math
   samples = samples.astype(np.float32)
   if samples.max() > 1.0:
       samples = samples / 32768.0  # Divide by max value
   ```

5. **Calculate RMS (Loudness)**
   ```python
   # RMS = sqrt(mean(samples²))
   # Square each sample, average them, take square root
   rms = np.sqrt(np.mean(samples**2))
   # Result: 0.0 (silent) to 1.0 (maximum loudness)
   ```

6. **Compare to Threshold**
   ```python
   # If RMS > 0.60, it's considered "loud"
   is_loud = rms > AUDIO_LOUD_THRESHOLD
   ```

7. **Save Flag if Loud**
   ```python
   if is_loud:
       # Save violation to database
       flags['loud_sound'] = [{
           'timestamp': datetime.utcnow().isoformat(),
           'message': f'Loud sound detected (RMS: {rms:.3f})'
       }]
   ```

**Real-World Analogy:**
Think of it like a noise meter:
- **Silent room**: RMS = 0.1 (very quiet)
- **Normal talking**: RMS = 0.4-0.6 (moderate)
- **Shouting**: RMS = 0.8-1.0 (very loud)
- **Threshold 0.60**: Anything above this triggers an alert

**Why Different Thresholds?**
- **0.60**: Very sensitive - catches background noise, whispers, normal talking
- **0.75**: Moderate - catches normal talking but not whispers
- **0.90**: Less sensitive - only catches shouting or very loud sounds

#### `POST /frontend_alert` (Lines 101-158)
**Purpose**: Receives browser-based cheating alerts from frontend

**Input**: JSON payload
```json
{
  "type": "tabSwitch" | "keyboardShortcut" | "rightClick" | 
          "fullscreenExit" | "captions",
  "user_id": "...",
  "session_id": "...",
  ...other fields
}
```

**Type Mapping** (Lines 93-99):
- `keyboardShortcut` → `keyboard_shortcut`
- `tabSwitch` → `tab_switch`
- `rightClick` → `right_click`
- `fullscreenExit` → `fullscreen_exit`
- `captions` → `captions`

**Process**:
1. Validates type and session_id
2. Creates flag entry with timestamp
3. Appends to Firestore `proctoring_sessions.flags.<flag_key>` using `ArrayUnion`
4. Creates session doc if doesn't exist

**Response**: `{"status": "ok", "stored_as": "flag_key"}`

**🚨 How It Works (Simple Explanation):**

**What is a Frontend Alert?**
The browser (frontend) can detect certain behaviors that the server can't see:
- Tab switching (Alt+Tab or clicking another tab)
- Keyboard shortcuts (Ctrl+C, Ctrl+V)
- Right-click attempts
- Exiting fullscreen mode
- Using captions/accessibility features

**Step-by-Step Process:**

1. **Browser Detects Violation**
   ```javascript
   // In browser JavaScript (templates/index.html)
   // User presses Ctrl+C (copy)
   document.addEventListener('keydown', (e) => {
       if (e.ctrlKey && e.key === 'c') {
           // Send alert to server
           fetch('/frontend_alert', {
               method: 'POST',
               body: JSON.stringify({
                   type: 'keyboardShortcut',
                   session_id: 'session_123',
                   user_id: 'user_456'
               })
           });
       }
   });
   ```

2. **Server Receives Alert**
   ```python
   # In detection.py
   data = request.get_json()
   alert_type = data.get("type")  # "keyboardShortcut"
   session_id = data.get("session_id")
   ```

3. **Map Type to Database Key**
   ```python
   # Convert frontend name to database name
   FRONTEND_TYPE_MAP = {
       "keyboardShortcut": "keyboard_shortcut",  # Add underscores
       "tabSwitch": "tab_switch",
       # etc.
   }
   flag_key = FRONTEND_TYPE_MAP.get(alert_type)
   ```

4. **Create Flag Entry**
   ```python
   flag_entry = {
       "timestamp": datetime.utcnow().isoformat(),  # When it happened
       "type": alert_type,  # What happened
       "user_id": user_id,  # Who did it
       "source": "frontend",  # Where it was detected
   }
   ```

5. **Append to Database**
   ```python
   # Add to existing flags array (don't overwrite)
   session_ref.update({
       f"flags.{flag_key}": firestore.ArrayUnion([flag_entry])
   })
   # ArrayUnion = add to array without removing existing items
   ```

**Real-World Analogy:**
Think of it like a security guard reporting incidents:
- **Browser** = Security guard watching the student
- **Alert** = Guard writes a report
- **Server** = Receives report and files it
- **Database** = Filing cabinet storing all reports

**Why ArrayUnion?**
If a student switches tabs 5 times, we want to keep all 5 records, not just the last one. ArrayUnion adds to the list without deleting previous entries.

---

### Session Management Endpoints

#### `POST /start_session` (Lines 1066-1101)
**Purpose**: Creates new proctoring session

**Input**: JSON
```json
{
  "student_id": "...",
  "exam_id": "..."
}
```

**Creates Firestore Document**:
```json
{
  "student_id": "...",
  "exam_id": "...",
  "start_time": "ISO timestamp",
  "status": "active",
  "flags": {},
  "video_path": null,
  "clip_path": null
}
```

**Response**: `{"success": true, "session_id": "...", "message": "..."}`

**Note**: Frontend uses `/add?collection=proctoring_sessions` instead (see templates/index.html:960)

#### `POST /end_session` (Lines 1104-1128)
**Purpose**: Ends active proctoring session

**Input**: JSON `{"session_id": "..."}`

**Updates**: Sets `end_time` and `status: "completed"`

#### `GET /get_session_status` (Lines 1131-1152)
**Purpose**: Retrieves session status

**Query Parameter**: `session_id`

**Response**: Full session document with `id` field

---

### Video Management

#### `POST /upload_video` (Lines 913-962)
**Purpose**: Uploads video recordings for proctoring sessions

**Query Parameters**:
- `session_id` (required): Links video to session
- `type` (optional): `"clip"` for cheating clips, `"full"` for full session (default)

**Input**: Multipart form-data with `video` file

**Process**:
1. Validates session_id and file
2. Generates secure filename: `session_{session_id}_{timestamp}_{original_name}`
3. Saves to `static/recordings/` directory
4. Updates Firestore:
   - `clip_path` if type="clip"
   - `video_path` if type="full"
   - `clip_uploaded_at` or `video_uploaded_at` timestamp

**Response**: `{"message": "uploaded", "path": "/static/recordings/..."}`

**Used by**: 
- `templates/index.html` (lines 1022, 2413) for session recordings
- Cheating clip recording (5-second clips on violation)

---

### Data Management Endpoints (CRUD)

#### `POST /add` (Lines 624-632)
**Purpose**: Generic document creation

**Query Parameter**: `collection` (default: `"users"`)

**Input**: JSON document data

**Response**: `{"id": "document_id"}`

**Used by**: Frontend for creating users, exams, classes, proctoring_sessions, exam_results

#### `GET /select` (Lines 635-658)
**Purpose**: Generic document query

**Query Parameters**:
- `collection` (default: `"users"`)
- `id` (optional): Get single document

**Behavior**:
- With `id`: Returns single document with `id` field
- Without `id`: Returns all documents in collection

**Used extensively**: All templates query Firestore collections

#### `POST /edit/<id>` (Lines 661-668)
**Purpose**: Update document

**Query Parameter**: `collection` (default: `"users"`)

**Input**: JSON with fields to update

**Response**: `{"message": "Updated"}`

#### `POST /delete/<id>` (Lines 671-678)
**Purpose**: Delete document

**Query Parameter**: `collection` (default: `"users"`)

**Response**: `{"message": "Deleted"}`

---

### Authentication & Routing

#### `POST /login` (Lines 681-702)
**Purpose**: User authentication

**Input**: JSON `{"email": "...", "password": "..."}`

**Process**:
1. Queries Firestore `users` collection
2. Matches email AND password (plaintext - security concern)
3. Returns user data with `id` field

**Response**:
```json
{
  "success": true,
  "user": {...user_data, "id": "..."}
}
```
or `{"success": false, "error": "Invalid credentials"}`

**Security Note**: Passwords stored in plaintext - should use hashing

#### Route Handlers (Lines 705-808)
- `GET /` → `login.html`
- `GET /detection` → `index.html`
- `GET /register` → `register.html`
- `GET /dashboard` → `dashboard.html` (admin)
- `GET /student_dashboard` → `student_dashboard.html`
- `GET /proctor_dashboard` → `proctor_dashboard.html`
- `GET /users` → `users.html`
- `GET /exams` → `exams.html`
- `GET /classes` → `classes.html`
- `GET /reports` → `reports.html`
- `GET /profile` → `profile.html`
- `GET /take_exam/<exam_id>` → `index.html` (with exam_id)
- `POST /logout` → JSON response

#### `GET /take_exam/<exam_id>` (Lines 750-793)
**Special Logic**:
1. Validates exam exists
2. Checks exam start/end dates
3. Validates student's `class_id` matches exam's `class_id`
4. Redirects with flash messages if validation fails
5. Renders `index.html` with `exam_id` context

---

### Report Sharing

#### `POST /share_report` (Lines 965-1007)
**Purpose**: Creates shareable link for reports

**Input**: JSON
```json
{
  "student_id": "...",
  "exam_id": "...",
  "session_id": "..."
}
```

**Process**:
1. Generates unique token (32 bytes, URL-safe)
2. Stores in `shared_reports` collection
3. Returns shareable URL

**Response**: `{"success": true, "share_url": "...", "token": "..."}`

#### `GET /shared_report/<token>` (Lines 1010-1063)
**Purpose**: Displays shared report (no authentication required)

**Process**:
1. Looks up token in `shared_reports` collection
2. Validates expiration (if set)
3. Fetches student, exam, and session data
4. Renders `shared_report.html` template

**Security**: Private access via token (no login required)

---

### Cleanup & Maintenance

#### `POST /delete_old_reports` (Lines 811-910)
**Purpose**: Manual cleanup of old proctoring reports

**Input**: JSON
```json
{
  "delete_all": true,  // Delete all reports
  // OR
  "cutoff_date": "ISO date string",  // Specific date
  // OR
  "cutoff_days": 14  // Days ago (default: 14)
}
```

**Process**:
1. Queries old `proctoring_sessions` (before cutoff or all)
2. Deletes associated video files from `static/recordings/`
3. Deletes session documents
4. If `delete_all`: Also deletes all `exam_results`
5. Cleans up expired `shared_reports`

**Response**: `{"success": true, "message": "Deleted X sessions and Y shares"}`

#### `cleanup_old_reports()` (Lines 1155-1208)
**Purpose**: Scheduled automatic cleanup (runs daily at 2 AM)

**Process**:
- Same as manual cleanup but hardcoded to 14 days
- Runs via APScheduler background scheduler

**Scheduler Setup** (Lines 1211-1224):
```python
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=cleanup_old_reports,
    trigger=CronTrigger(hour=2, minute=0),  # Daily at 2 AM
    id='cleanup_old_reports',
    name='Clean up old proctoring reports',
    replace_existing=True
)
scheduler.start()
```

**Shutdown**: Registered with `atexit` for graceful shutdown

---

## Model Initialization

### MediaPipe Models (Lines 202-213)
**Loaded Once at Startup** (not per request - critical for memory):

1. **Holistic Model**:
```python
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False  # Video mode for better performance
)
```

2. **Face Detection Model**:
```python
face_detection_model = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 = short-range (faster), 1 = full-range
    min_detection_confidence=0.4
)
```

**Thread Safety**: Models are thread-safe and reused across requests

**🤖 How AI Models Work (Simple Explanation):**

**What is an AI Model?**
Think of an AI model like a trained expert:
- **Training**: The model was "taught" on millions of images/videos
- **Knowledge**: It learned patterns (what faces look like, what phones look like)
- **Usage**: When you give it a new image, it uses its knowledge to identify things

**Why Load Once at Startup?**

**Bad Way (Loading Every Time):**
```python
# DON'T DO THIS - Very slow!
@app.route('/predict')
def predict():
    model = mp_holistic.Holistic()  # Load model (takes 2-3 seconds!)
    results = model.process(image)
    # Every request waits 2-3 seconds just to load the model
```

**Good Way (Load Once):**
```python
# Load models when server starts (once)
holistic_model = mp_holistic.Holistic()  # Takes 2-3 seconds ONCE

@app.route('/predict')
def predict():
    results = holistic_model.process(image)  # Instant! Model already loaded
    # Reuse the same model for all requests
```

**Analogy:** Like keeping tools ready on your workbench instead of going to the toolbox every time.

**Model Parameters Explained:**

**1. Holistic Model:**
```python
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,  # Must be 50% sure to detect
    min_tracking_confidence=0.5,  # Must be 50% sure to track
    static_image_mode=False  # Video mode (faster for videos)
)
```
- **min_detection_confidence**: How sure the AI must be before saying "I found something"
  - 0.5 = 50% sure (balanced)
  - 0.9 = 90% sure (very strict, might miss things)
  - 0.3 = 30% sure (very loose, might have false positives)
- **min_tracking_confidence**: Once detected, how sure to keep tracking it
- **static_image_mode**: 
  - `False` = Video mode (optimized for continuous frames)
  - `True` = Image mode (better for single photos, slower)

**2. Face Detection Model:**
```python
face_detection_model = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 = short-range, 1 = full-range
    min_detection_confidence=0.4
)
```
- **model_selection**:
  - `0` = Short-range (faces close to camera, faster)
  - `1` = Full-range (faces far away, slower but more accurate)
- **min_detection_confidence**: 0.4 = 40% confidence threshold

**How Models Are Used:**

**Step 1: Model Loads (At Startup)**
```python
# This runs ONCE when server starts
print("Loading AI models...")
holistic_model = mp_holistic.Holistic(...)  # Takes 2-3 seconds
face_detection_model = mp_face_detection.FaceDetection(...)  # Takes 1 second
print("Models loaded!")
```

**Step 2: Model Used (During Requests)**
```python
# This runs for EVERY request (but model already loaded)
@app.route('/predict')
def predict():
    # Use the pre-loaded model (instant!)
    results = holistic_model.process(image)
    # Process results...
```

**Memory Considerations:**
- Each model uses ~50-100 MB of RAM
- Loading once = Models stay in memory, ready to use
- Loading every time = Would use same memory but waste time reloading

**Thread Safety:**
- "Thread-safe" means multiple requests can use the same model simultaneously
- Like multiple people using the same tool at the same time (safely)
- MediaPipe models are designed for this

### YOLO Model (Lines 187-200)
**Optional Object Detection**:
- Path: `models/yolov5s.onnx`
- Loaded via OpenCV DNN
- Backend: OpenCV DNN (CPU)
- Target: CPU (for Render free tier compatibility)
- **Note**: User requested exclusion of yolov9 analysis

**Fallback**: If model not found, `yolo_net = None` and object detection skipped

---

## Firebase Integration

### Initialization (Lines 160-164)
```python
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_admin_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
```

**Global Database Client**: `db` used throughout application

**🗄️ How Firebase Works (Simple Explanation):**

**What is Firebase?**
Firebase is Google's cloud database service. Think of it like a giant filing cabinet in the cloud that stores all your data.

**How It Connects:**
1. **Credentials File**: `firebase_admin_key.json` is like a key card that proves you're allowed to access the database
2. **Initialize**: Connect to Firebase using the credentials
3. **Database Client**: `db` is like a remote control to access the database

**In Simple Terms:**
```python
# Step 1: Check if we're already connected (don't connect twice!)
if not firebase_admin._apps:
    # Step 2: Load the key card (credentials file)
    cred = credentials.Certificate("firebase_admin_key.json")
    
    # Step 3: Use key card to enter Firebase
    firebase_admin.initialize_app(cred)
    
    # Step 4: Get the remote control (database client)
    db = firestore.client()
    
# Now we can use 'db' anywhere in the code to read/write data
```

**Why Check `if not firebase_admin._apps`?**
If the app restarts or reloads, we don't want to connect twice. This check prevents errors.

**How Data is Stored:**
Firebase uses a structure like:
```
Firebase (Cloud)
├── Collection: "users"
│   ├── Document: "user_123"
│   │   ├── Field: "name" = "John"
│   │   ├── Field: "email" = "john@example.com"
│   │   └── Field: "role" = "student"
│   └── Document: "user_456"
│       └── ...
├── Collection: "exams"
│   └── ...
└── Collection: "proctoring_sessions"
    └── ...
```

**Reading Data Example:**
```python
# Get a user document
user_ref = db.collection('users').document('user_123')
user_doc = user_ref.get()
user_data = user_doc.to_dict()
# Result: {"name": "John", "email": "john@example.com", ...}
```

**Writing Data Example:**
```python
# Add a new document
db.collection('users').add({
    'name': 'Jane',
    'email': 'jane@example.com',
    'role': 'student'
})
# Firebase automatically creates a unique ID
```

**Updating Data Example:**
```python
# Update existing document
db.collection('users').document('user_123').update({
    'name': 'John Smith'  # Only updates this field
})
```

### Collections Used
1. **users**: User accounts (students, proctors, admins)
2. **exams**: Exam definitions
3. **classes**: Class/section management
4. **proctoring_sessions**: Active/completed proctoring sessions
5. **exam_results**: Student exam scores and answers
6. **shared_reports**: Shareable report tokens

### Session Document Structure
```json
{
  "student_id": "...",
  "exam_id": "...",
  "start_time": "ISO timestamp",
  "end_time": "ISO timestamp (optional)",
  "status": "active" | "completed" | "terminated",
  "flags": {
    "twofaces": [{timestamp, message, ...}],
    "loud_sound": [{timestamp, message, ...}],
    "keyboard_shortcut": [{timestamp, type, source: "frontend", ...}],
    "tab_switch": [...],
    "right_click": [...],
    "fullscreen_exit": [...],
    "captions": [...],
    "object_detected_cell_phone": [...],
    ...
  },
  "video_path": "/static/recordings/...",
  "clip_path": "/static/recordings/...",
  "video_uploaded_at": "ISO timestamp",
  "clip_uploaded_at": "ISO timestamp"
}
```

---

## Frontend Integration

### Templates Using Detection.py

#### `templates/index.html` (Main Exam Interface)
**Key Interactions**:

1. **Session Creation** (Line ~960):
   - Uses `/add?collection=proctoring_sessions` (not `/start_session`)
   - Creates session with initial flags structure

2. **Video Frame Processing** (Line ~2019):
   - Calls `/predict?session_id=...` every 15 seconds (configurable)
   - Sends processed image from client-side MediaPipe
   - Server-side processing is optional (can disable with `intervalMs = 0`)

3. **Video Upload** (Lines ~1022, ~2413):
   - Full session recording: `/upload_video?session_id=...`
   - Cheating clips: `/upload_video?session_id=...&type=clip`
   - 5-second clips recorded on violation detection

4. **Session End** (Line ~909):
   - Updates session via `/edit/{session_id}?collection=proctoring_sessions`
   - Sets `end_time` and `status: "completed"`

**Note**: Frontend doesn't use `/frontend_alert` endpoint - instead directly updates Firestore via `/edit`

**🌐 How Frontend and Backend Connect (Simple Explanation):**

**What is Frontend vs Backend?**
- **Frontend** = What the user sees (browser, HTML, JavaScript)
- **Backend** = What the server does (Python, Flask, database)

**How They Communicate:**

**1. Session Creation Flow:**
```javascript
// Frontend (JavaScript in browser)
async function initAlertsSession() {
    // Step 1: Prepare data
    const sessionData = {
        user_id: examState.userId,
        exam_id: examState.examId,
        start_time: new Date().toISOString(),
        status: 'active',
        flags: {}  // Empty flags object
    };
    
    // Step 2: Send HTTP POST request to backend
    const response = await fetch('/add?collection=proctoring_sessions', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(sessionData)
    });
    
    // Step 3: Wait for response
    const data = await response.json();
    alertsSession.sessionId = data.id;  // Save session ID
}
```

```python
# Backend (Python Flask)
@app.route('/add', methods=['POST'])
def add():
    collection = request.args.get('collection', 'users')
    data = request.json  # Get data from frontend
    
    # Add to Firebase
    doc_ref = db.collection(collection).add(data)
    
    # Send back the new document ID
    return jsonify({'id': doc_ref[1].id})
```

**2. Video Frame Processing Flow:**
```javascript
// Frontend: Every 15 seconds
async function sendPreviewToServer() {
    // Step 1: Get current video frame (image)
    const canvas = document.getElementById('videoCanvas');
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Step 2: Convert to blob (file-like object)
    const blob = await (await fetch(imageData)).blob();
    
    // Step 3: Create form data
    const form = new FormData();
    form.append('frame', blob, 'client.jpg');
    
    // Step 4: Send to backend
    await fetch(`/predict?session_id=${sessionId}`, {
        method: 'POST',
        body: form
    });
    // Backend processes and saves flags automatically
}
```

```python
# Backend: Receives frame and processes
@app.route('/predict', methods=['POST'])
def predict():
    # Step 1: Get image from request
    file = request.files['frame']
    img = cv2.imdecode(file.read(), cv2.IMREAD_COLOR)
    
    # Step 2: Process (detect faces, objects, etc.)
    face_count = count_faces_bounding_boxes(img)
    # ... more processing ...
    
    # Step 3: Save flags to Firebase
    if session_id:
        db.collection('proctoring_sessions').document(session_id).update({
            'flags': flags
        })
    
    # Step 4: Return results
    return jsonify({
        'face_count': face_count,
        'looking_at_screen': looking,
        # ... etc
    })
```

**3. Video Upload Flow:**
```javascript
// Frontend: When exam ends
window.mediaRecorder.onstop = async () => {
    // Step 1: Combine video chunks into one file
    const blob = new Blob(window.mediaChunks, { type: 'video/webm' });
    
    // Step 2: Create form data
    const formData = new FormData();
    formData.append('video', blob, `session_${sessionId}.webm`);
    
    // Step 3: Upload to backend
    await fetch(`/upload_video?session_id=${sessionId}`, {
        method: 'POST',
        body: formData
    });
}
```

```python
# Backend: Saves video file
@app.route('/upload_video', methods=['POST'])
def upload_video():
    # Step 1: Get video file
    video_file = request.files['video']
    session_id = request.args.get('session_id')
    
    # Step 2: Generate safe filename
    filename = f"session_{session_id}_{timestamp}.webm"
    
    # Step 3: Save to disk
    save_path = os.path.join('static', 'recordings', filename)
    video_file.save(save_path)
    
    # Step 4: Update database with file path
    db.collection('proctoring_sessions').document(session_id).update({
        'video_path': f'/static/recordings/{filename}'
    })
    
    return jsonify({'message': 'uploaded', 'path': save_path})
```

**Key Concepts:**

**1. Asynchronous (async/await):**
- Frontend uses `async/await` because network requests take time
- `await` means "wait for this to finish before continuing"
- Prevents the page from freezing while waiting

**2. JSON (JavaScript Object Notation):**
- Format for sending data between frontend and backend
- Looks like: `{"name": "John", "age": 25}`
- Both JavaScript and Python can read/write JSON

**3. FormData:**
- Used for file uploads (videos, images)
- Can't send files as JSON, need FormData instead

**4. Fetch API:**
- Modern way to send HTTP requests from JavaScript
- Replaces older XMLHttpRequest
- Returns a Promise (like a "future result")

**Real-World Analogy:**
Think of frontend-backend communication like ordering food:
- **Frontend** = Customer calling restaurant
- **HTTP Request** = Phone call with order
- **Backend** = Restaurant kitchen
- **HTTP Response** = Food delivery
- **Database** = Restaurant's inventory/storage

#### `templates/take_exam.html`
- Redirects to `/take_exam/<exam_id>` route
- Validates exam access and dates

#### `templates/reports.html`
- Queries `proctoring_sessions` via `/select?collection=proctoring_sessions`
- Displays flags, video recordings, timeline
- Uses `/share_report` to create shareable links

---

## Configuration & Constants

### Audio Detection (Line 33)
```python
AUDIO_LOUD_THRESHOLD = 0.60  # RMS threshold (0.0 to 1.0)
```

### Face Detection Confidence (Line 211)
```python
min_detection_confidence=0.4
```

### Head Pose Thresholds (Lines 279-280)
```python
looking = (abs(yaw) < 30) and (abs(pitch) < 20)
```

### YOLO Confidence Threshold (Line 507)
```python
confidence > 0.5
```

### Flask Secret Key (Line 87)
```python
app.secret_key = 'your-secret-key-here'  # Should be changed in production
```

---

## Memory Optimization Strategies

### For Render Free Tier (512 MB RAM limit)

1. **Model Loading**: Models loaded once at startup (Lines 191-213)
2. **Frame Resizing**: YOLO processes 640×640 max (Lines 484-492)
3. **Low Processing Frequency**: Frontend sends frames every 15 seconds (not every frame)
4. **Client-Side Processing**: Most detection runs in browser (MediaPipe Face Mesh, COCO-SSD)
5. **Timeout Handling**: 25-second timeout on `/predict` calls (templates/index.html:2016)

### Performance Notes
- **1-5 students**: Works smoothly
- **5-10 students**: May experience delays
- **10+ students**: Consider paid tier or disable server-side processing

---

## Security Considerations

### Current Issues

1. **Plaintext Passwords**: `/login` endpoint compares plaintext passwords (Line 688)
   - **Recommendation**: Use bcrypt or similar hashing

2. **Secret Key**: Hardcoded default secret key (Line 87)
   - **Recommendation**: Use environment variable

3. **No Rate Limiting**: Endpoints lack rate limiting
   - **Recommendation**: Add Flask-Limiter

4. **No Authentication Middleware**: Routes don't verify user sessions
   - **Recommendation**: Add session-based authentication

5. **File Upload Security**: Basic filename sanitization (Line 934)
   - **Recommendation**: Validate file types, scan for malware

### Positive Security Features

1. **Secure Filenames**: Uses `secure_filename()` (Line 934)
2. **Session Validation**: Exam access validates class_id (Lines 777-791)
3. **Date Validation**: Exam start/end dates checked (Lines 763-772)

---

## Error Handling

### Exception Handling Patterns

1. **Try-Except Blocks**: Most endpoints wrapped in try-except
2. **Silent Failures**: Some failures logged but don't break flow (e.g., Line 603)
3. **Error Responses**: Returns JSON error responses with status codes
4. **Logging**: Uses `print()` statements (should use proper logging)

### Common Error Scenarios

- **Missing Session**: Creates session if doesn't exist (Line 142)
- **Model Loading Failure**: Gracefully handles missing YOLO model (Line 199)
- **Audio Decode Failure**: Falls back to raw PCM (Line 329)
- **Image Decode Failure**: Returns 400 error (Line 409)

---

## Deployment Configuration

### SSL/HTTPS Support (Lines 1227-1238)
```python
if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=(ssl_cert, ssl_key))
else:
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Note**: Camera access requires HTTPS in browsers

### Production Mode
- Uses Gunicorn (see `Procfile`)
- Command: `gunicorn detection:app`
- Debug mode disabled in production

---

## Dependencies Summary

From `requirements.txt`:
- Flask==3.0.3
- Werkzeug==3.0.3
- firebase-admin==6.4.0
- google-cloud-firestore==2.16.0
- opencv-python-headless==4.9.0.80
- mediapipe==0.10.9
- protobuf==3.20.3
- numpy==1.26.4
- librosa==0.10.2.post1
- soundfile==0.12.1
- audioread==3.0.1
- pydub==0.25.1
- APScheduler==3.10.4
- gunicorn==21.2.0
- requests==2.31.0

---

## Key Workflows

### Exam Proctoring Flow

1. **Student starts exam** → Frontend calls `/add?collection=proctoring_sessions`
2. **During exam**:
   - Client-side detection (MediaPipe, COCO-SSD) runs continuously
   - Every 15 seconds: Frame sent to `/predict` for server-side validation
   - Violations logged via `/edit` endpoint (updates `flags`)
   - Video recording uploaded via `/upload_video`
3. **On violation**: 5-second clip recorded and uploaded as `clip`
4. **Exam ends** → Frontend calls `/edit` to set `status: "completed"`

### Audio Detection Flow

1. **Frontend captures audio** → Analyzes client-side (RMS calculation)
2. **Optional**: Sends to `/detect_audio` for server-side validation
3. **If loud**: Flags stored in `proctoring_sessions.flags.loud_sound`

### Report Generation Flow

1. **Proctor views reports** → `/select?collection=proctoring_sessions`
2. **Filters by student/exam** → Frontend JavaScript filtering
3. **Shares report** → `/share_report` creates token
4. **Access shared report** → `/shared_report/<token>` (no auth required)

---

## Code Quality Notes

### Strengths
- Memory-efficient model loading
- Flexible input handling (multiple formats)
- Comprehensive error handling
- Scheduled cleanup tasks
- Good separation of concerns

### Areas for Improvement
- Replace `print()` with proper logging
- Add authentication middleware
- Implement password hashing
- Add rate limiting
- Add input validation/sanitization
- Add unit tests
- Add API documentation (Swagger/OpenAPI)

---

## Summary

`detection.py` is a comprehensive Flask application that provides:
- **Real-time cheating detection** via computer vision and audio analysis
- **Session management** for proctored exams
- **Data persistence** via Firebase Firestore
- **Report generation** and sharing capabilities
- **Memory-optimized** for free-tier hosting

The application is production-ready but would benefit from security hardening (authentication, password hashing, rate limiting) and improved error handling/logging.

---

## 🎓 Complete Beginner's Guide: How Everything Works Together

### The Big Picture Flow

**When a Student Takes an Exam:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. STUDENT CLICKS "START EXAM"                               │
│    Browser (Frontend) sends: POST /add?collection=...       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FLASK RECEIVES REQUEST                                     │
│    - Reads the request data                                  │
│    - Creates session in Firebase                             │
│    - Returns session_id to browser                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. BROWSER STARTS CAMERA & MICROPHONE                        │
│    - Gets permission from student                            │
│    - Starts video stream                                     │
│    - Starts audio recording                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CLIENT-SIDE DETECTION (Every Frame)                       │
│    - MediaPipe analyzes video in browser                     │
│    - Detects faces, hands, objects                            │
│    - Shows alerts immediately                                 │
│    - Logs violations to session                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. SERVER-SIDE VALIDATION (Every 15 Seconds)                │
│    Browser sends frame → POST /predict                       │
│    Flask processes → Detects violations                      │
│    Saves flags → Firebase database                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. VIOLATION DETECTED                                         │
│    - Browser shows alert                                     │
│    - Records 5-second clip                                    │
│    - Uploads clip → POST /upload_video?type=clip             │
│    - Updates violation count                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. EXAM ENDS                                                  │
│    - Browser stops recording                                 │
│    - Uploads full video → POST /upload_video                │
│    - Updates session → POST /edit (status: "completed")      │
│    - Saves exam answers → POST /add?collection=exam_results  │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts Explained Simply

**1. HTTP Requests (Like Sending Letters)**
- **GET**: "Give me information" (like asking for a webpage)
- **POST**: "Here's data, do something with it" (like submitting a form)
- **Response**: Server sends back data (like a reply letter)

**2. JSON (Data Format)**
- Like a structured note that both browser and server understand
- Example: `{"name": "John", "age": 25}`
- Easy to read and write for both humans and computers

**3. Async/Await (Waiting for Responses)**
- Network requests take time (like waiting for mail)
- `await` = "Wait here until this finishes"
- Prevents code from continuing before getting a response

**4. Database (Firebase)**
- Like a filing cabinet in the cloud
- Stores all data (users, exams, sessions, flags)
- Can read and write from anywhere

**5. AI Models (MediaPipe, YOLO)**
- Pre-trained experts that can recognize things in images
- Loaded once at startup (like having tools ready)
- Used repeatedly for all requests (like reusing tools)

**6. Client-Side vs Server-Side**
- **Client-Side**: Runs in browser (fast, immediate feedback)
- **Server-Side**: Runs on server (more powerful, can save to database)
- Both work together for best results

### Common Questions Answered

**Q: Why does it take time to load?**
A: AI models need to load into memory first (like loading a program). Once loaded, they're fast.

**Q: How does it detect cheating?**
A: Multiple methods:
- Face detection (counts faces)
- Head pose (checks if looking at screen)
- Object detection (finds phones, books)
- Audio analysis (detects loud sounds)
- Browser events (tab switches, keyboard shortcuts)

**Q: Where is data stored?**
A: Firebase Firestore (cloud database). Data persists even if server restarts.

**Q: Can students cheat?**
A: No system is 100% foolproof, but this system detects:
- Multiple people in frame
- Looking away from screen
- Prohibited objects (phones, books)
- Tab switching
- Keyboard shortcuts
- Loud sounds (talking to someone)

**Q: What happens if server crashes?**
A: 
- Data in Firebase is safe (cloud storage)
- Videos might be lost if not uploaded yet
- Session flags are saved immediately, so violations are recorded

**Q: How does it work on slow internet?**
A: 
- Client-side detection works offline (in browser)
- Server-side validation might be delayed
- Videos upload when exam ends (doesn't block exam)

### Learning Path for Beginners

**If you want to understand this code:**

1. **Start with Flask basics**
   - Learn what `@app.route()` does
   - Understand HTTP requests/responses
   - Practice with simple Flask apps

2. **Learn about databases**
   - Understand what Firebase/Firestore is
   - Learn how to read/write data
   - Practice with simple CRUD operations

3. **Understand HTTP communication**
   - Learn about GET vs POST
   - Understand JSON format
   - Practice with fetch() API

4. **Learn about AI/ML basics**
   - Understand what MediaPipe does
   - Learn about computer vision basics
   - Practice with simple image processing

5. **Put it all together**
   - Follow the code flow
   - Trace a request from browser to database
   - Modify small parts to see what happens

**Recommended Resources:**
- Flask documentation: https://flask.palletsprojects.com/
- Firebase documentation: https://firebase.google.com/docs
- MediaPipe documentation: https://mediapipe.dev/
- JavaScript fetch API: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API

