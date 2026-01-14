# Detection.py - Simplified Explanation

## What Is This System?

This is a **proctoring system** - like a digital exam supervisor that watches students taking online exams to detect cheating. Think of it as a security guard that never sleeps, watching through the camera and microphone to catch suspicious behavior.

---

## The Big Picture: How Everything Works Together

### The Three Main Parts

1. **The Browser (Frontend)**
   - What the student sees and interacts with
   - Captures video from webcam and audio from microphone
   - Detects some cheating behaviors immediately (like tab switching)
   - Sends data to the server for analysis

2. **The Server (Backend - detection.py)**
   - The "brain" that processes everything
   - Analyzes video frames for faces, objects, and suspicious behavior
   - Analyzes audio for loud sounds
   - Stores all violations in a database
   - Manages exam sessions

3. **The Database (Firebase)**
   - The "memory" that stores everything
   - Keeps track of users, exams, sessions, and violations
   - Like a filing cabinet that never gets full

---

## Key Features and How They Work

### 1. Face Detection

**What it does:** Counts how many faces are visible in the camera.

**Why it matters:** If there are 2 or more faces, someone else might be helping the student cheat.

**How it works:**
- Uses AI (MediaPipe) that was trained on millions of face images
- Scans each video frame
- Draws an imaginary box around each face found
- Reports the count back to the system

**Real-world analogy:** Like a bouncer at a club counting how many people are at the door.

---

### 2. Head Pose Detection (Gaze Tracking)

**What it does:** Determines if the student is looking at the screen or looking away.

**Why it matters:** If they're looking away, they might be looking at notes, a phone, or talking to someone.

**How it works:**
- Uses 6 key points on the face (nose tip, chin, eye corners, mouth corners)
- Calculates head rotation angles (left/right, up/down, tilt)
- If head is turned more than 30° left/right or 20° up/down, flags as "not looking"

**Real-world analogy:** Like checking if someone is making eye contact with you or looking elsewhere.

---

### 3. Object Detection

**What it does:** Scans the video for prohibited objects like phones, laptops, books, etc.

**Why it matters:** Students shouldn't have reference materials or communication devices during exams.

**How it works:**
- Uses YOLO AI model trained to recognize 80+ different objects
- Scans each frame for objects
- If it finds a phone, book, or laptop, it immediately flags it
- Phone detection = instant exam termination (most serious violation)

**Real-world analogy:** Like airport security scanning for prohibited items.

---

### 4. Audio Analysis

**What it does:** Monitors microphone input for loud sounds or talking.

**Why it matters:** Loud sounds might indicate the student is talking to someone or receiving help.

**How it works:**
- Measures sound "loudness" using RMS (Root Mean Square)
- Compares against a threshold (60% of maximum volume)
- If sound exceeds threshold, flags as violation
- Threshold can be adjusted: lower = more sensitive, higher = less sensitive

**Real-world analogy:** Like a noise meter that alerts when sound gets too loud.

---

### 5. Browser Behavior Detection

**What it does:** Monitors what the student does in the browser itself.

**Detects:**
- **Tab switching:** Student switches to another browser tab
- **Keyboard shortcuts:** Copy (Ctrl+C), Paste (Ctrl+V), etc.
- **Right-click:** Attempts to access context menu
- **Fullscreen exit:** Student exits fullscreen mode
- **Window resize:** Student tries to resize the exam window

**Why it matters:** These actions suggest the student might be accessing notes, copying questions, or using other applications.

**How it works:**
- Browser JavaScript detects these events immediately
- Sends alerts to server
- Server logs them as violations
- Multiple violations lead to warnings, then exam termination

**Real-world analogy:** Like a security guard watching for suspicious movements.

---

## The Exam Flow: Step by Step

### Step 1: Student Starts Exam
- Student clicks "Start Exam" button
- System creates a "proctoring session" - like opening a case file
- Session gets a unique ID number
- Camera and microphone permissions requested

### Step 2: During the Exam
- **Continuous monitoring:** Camera and microphone record everything
- **Client-side detection:** Browser analyzes video/audio in real-time (fast, immediate alerts)
- **Server-side validation:** Every 15 seconds, sends a frame to server for deeper analysis
- **Violation logging:** Every suspicious behavior is logged with timestamp

### Step 3: When Violation Detected
- **Alert shown:** Student sees warning on screen
- **Clip recorded:** System records 5-second video clip of the violation
- **Flag saved:** Violation saved to database with details
- **Strike counted:** System tracks how many violations (3 strikes = termination)

### Step 4: Exam Ends
- **Full video uploaded:** Complete exam recording saved
- **Session closed:** Proctoring session marked as "completed"
- **Results saved:** Exam answers and scores stored
- **Report generated:** Proctor can review all violations and video

---

## How Data Flows Through the System

### The Request-Response Cycle

1. **Browser sends request:** "Hey server, analyze this video frame"
2. **Server receives:** Gets the video frame data
3. **Server processes:** Runs AI models to detect violations
4. **Server saves:** Stores results in database
5. **Server responds:** Sends back what was found
6. **Browser displays:** Shows results to student/proctor

**Analogy:** Like ordering food at a restaurant:
- You (browser) place order (send request)
- Kitchen (server) prepares food (processes)
- Kitchen stores order in system (saves to database)
- Waiter brings food (sends response)
- You receive food (browser displays results)

---

## Important Concepts Explained Simply

### What is Flask?
Flask is like a waiter in a restaurant. It receives orders (requests) from customers (browsers), takes them to the kitchen (your code), and brings back the food (responses). Each URL like `/predict` or `/login` is like a different menu item.

### What is Firebase?
Firebase is Google's cloud database - like a filing cabinet in the sky. It stores all your data (users, exams, violations) and you can access it from anywhere. It's organized into "collections" (like drawers) and "documents" (like files in those drawers).

### What are AI Models?
AI models are like trained experts. They were "taught" by looking at millions of examples (images, videos). Once trained, they can recognize patterns - like identifying faces, objects, or suspicious behavior. The models used here are:
- **MediaPipe:** Detects faces, hands, and body pose
- **YOLO:** Detects objects (phones, books, etc.)

### Client-Side vs Server-Side
- **Client-Side:** Runs in the browser (fast, immediate, but limited)
- **Server-Side:** Runs on the server (more powerful, can save to database, but slower)

Both work together: client-side gives immediate feedback, server-side does deep analysis and saves everything.

### Why Load Models Once?
AI models are heavy (like heavy tools). Loading them takes 2-3 seconds. If you loaded them every time someone made a request, everything would be slow. Instead, they're loaded once when the server starts, then reused for all requests (like keeping tools ready on your workbench).

---

## Violation System

### How Violations Work

Most violations follow a **3-strike system:**
1. **First violation:** Warning shown, exam paused for 5 seconds
2. **Second violation:** Another warning, exam paused again
3. **Third violation:** Exam terminated immediately, score set to 0

**Exception - Phone Detection:**
- **First detection:** Immediate termination (no warnings)
- Most serious violation because phones are the easiest way to cheat

### Types of Violations

| Violation Type | What It Detects | Strike Limit |
|---------------|----------------|--------------|
| Multiple Faces | 2+ people in camera | 3 strikes |
| Phone Detection | Cell phone visible | 1 strike (instant termination) |
| Loud Sound | Talking or loud noise | 3 strikes |
| Tab Switch | Switching browser tabs | 3 strikes |
| Keyboard Shortcut | Copy/paste shortcuts | 3 strikes |
| Right Click | Context menu access | 3 strikes |
| Fullscreen Exit | Exiting fullscreen mode | 3 strikes |
| Face Absent | No face detected | 3 strikes |
| Looking Away | Head turned away from screen | Logged but not counted |

---

## Session Management

### What is a Proctoring Session?

A session is like a case file for one student taking one exam. It contains:
- **Who:** Student ID and exam ID
- **When:** Start time and end time
- **What happened:** All violations/flags with timestamps
- **Evidence:** Links to video recordings (full session + violation clips)
- **Status:** Active, completed, or terminated

### Session Lifecycle

1. **Created:** When student starts exam
2. **Active:** During exam, violations are logged
3. **Completed:** Exam finished normally
4. **Terminated:** Exam ended due to violations

---

## Video Recording System

### Two Types of Recordings

1. **Full Session Recording:**
   - Records entire exam from start to finish
   - Uploaded when exam ends
   - Used for review and evidence

2. **Violation Clips:**
   - 5-second clips recorded when violation detected
   - Uploaded immediately
   - Quick reference for specific incidents

### Why Record Videos?

- **Evidence:** Proctors can review what actually happened
- **Review:** Helps determine if violations were intentional or accidental
- **Accountability:** Students know they're being recorded
- **Appeals:** Students can dispute false positives

---

## Memory Optimization (Why It Matters)

### The Challenge

Free hosting services (like Render free tier) have limited memory (512 MB). AI models are heavy and could crash the server if not optimized.

### Solutions Used

1. **Load models once:** Models loaded at startup, reused for all requests
2. **Resize images:** Process smaller images (640×640) instead of full resolution
3. **Low frequency:** Only send frames to server every 15 seconds (not every frame)
4. **Client-side processing:** Most detection happens in browser (doesn't use server memory)
5. **Timeout protection:** Requests timeout after 25 seconds to prevent hanging

**Result:** System can handle 1-5 students smoothly, 5-10 with occasional delays.

---

## Security Features

### What Protects the Exam

1. **Fullscreen requirement:** Student must stay in fullscreen mode
2. **Tab switch detection:** Alerts if student switches tabs
3. **Copy/paste prevention:** Blocks keyboard shortcuts
4. **Right-click disabled:** Prevents accessing context menu
5. **Window resize detection:** Alerts if student tries to resize
6. **Multiple face detection:** Catches if someone else enters frame
7. **Object detection:** Finds prohibited items
8. **Audio monitoring:** Detects talking or loud sounds

### What Happens When Violations Occur

- **Immediate alert:** Student sees warning on screen
- **Exam pause:** 5-second pause (except on 3rd strike)
- **Violation logged:** Saved to database with timestamp
- **Video clip:** 5-second clip recorded as evidence
- **Strike count:** System tracks violations
- **Termination:** After 3 strikes (or 1 for phone), exam ends with 0 score

---

## Report Generation

### What Proctors See

When a proctor reviews a student's exam session, they see:

1. **Session Summary:**
   - Start and end times
   - Duration
   - Final status (completed/terminated)

2. **Alert Statistics:**
   - Count of each violation type
   - Total violations
   - Most common violation

3. **Detailed Timeline:**
   - Chronological list of all violations
   - Timestamp for each violation
   - Description of what happened

4. **Video Evidence:**
   - Full session recording
   - Individual violation clips
   - Can play/pause to review

5. **Shareable Reports:**
   - Generate link to share report
   - No login required (private token access)
   - Useful for showing to administrators or parents

---

## Common Scenarios Explained

### Scenario 1: Student Looks at Phone

**What happens:**
1. Object detection AI sees phone in frame
2. Immediately flags as violation
3. Exam terminates (phone = instant termination)
4. 5-second clip recorded
5. Score set to 0
6. Proctor sees "Phone detected" in report

### Scenario 2: Student Switches Tabs

**What happens:**
1. Browser detects tab switch
2. Sends alert to server
3. First time: Warning shown, exam pauses 5 seconds
4. Violation logged with timestamp
5. Student continues exam
6. If happens 2 more times: Exam terminates

### Scenario 3: Someone Enters Room

**What happens:**
1. Face detection sees 2 faces
2. Flags as "multiple faces" violation
3. Warning shown, exam pauses
4. If person leaves: Student can continue
5. If happens 2 more times: Exam terminates

### Scenario 4: Student Talks to Someone

**What happens:**
1. Audio analysis detects loud sound (talking)
2. Flags as "loud sound" violation
3. Warning shown
4. If continues: More violations logged
5. After 3 violations: Exam terminates

---

## Why This System Works

### Advantages

1. **Multiple detection methods:** Catches different types of cheating
2. **Real-time monitoring:** Immediate alerts, not just after exam
3. **Evidence collection:** Videos provide proof of violations
4. **Scalable:** Can monitor many students simultaneously
5. **Automated:** Reduces need for human proctors
6. **Fair:** Consistent rules applied to all students

### Limitations

1. **Not 100% foolproof:** Determined cheaters might find ways around it
2. **False positives:** Legitimate actions might trigger alerts
3. **Privacy concerns:** Constant video/audio recording
4. **Technical requirements:** Needs good internet and working camera/mic
5. **Resource intensive:** Requires significant server resources for many students

---

## Key Takeaways

1. **Three-part system:** Browser captures, server analyzes, database stores
2. **Multiple detection methods:** Face detection, object detection, audio analysis, browser monitoring
3. **3-strike system:** Most violations allow 3 strikes before termination
4. **Phone detection is instant termination:** Most serious violation
5. **Everything is logged:** All violations saved with timestamps and video evidence
6. **Optimized for performance:** Models loaded once, images resized, low processing frequency
7. **Designed for fairness:** Consistent rules, evidence-based, reviewable

---

## How to Think About This System

Think of it like a **digital exam hall** with:
- **Security cameras** (video detection)
- **Microphones** (audio detection)
- **Guards watching** (AI models analyzing)
- **Rule enforcement** (violation system)
- **Evidence collection** (video recordings)
- **Report generation** (for proctors to review)

The goal is to create a fair, secure exam environment where students can't cheat, and if they try, there's clear evidence of what happened.

---

## For Non-Technical People

If you're not a programmer but need to understand this system:

**Think of it like this:**
- The browser is like a security camera that watches the student
- The server is like a security guard analyzing the footage
- The database is like a filing cabinet storing all the evidence
- The AI models are like trained experts who can spot suspicious behavior
- The violation system is like a three-strike policy at work

**The flow:**
1. Student takes exam → Camera watches
2. Suspicious behavior detected → Alert triggered
3. Evidence saved → Video clip recorded
4. Violations counted → After 3 strikes, exam ends
5. Proctor reviews → Sees all violations and evidence

**The goal:** Make online exams as secure as in-person exams, with clear evidence of any cheating attempts.

