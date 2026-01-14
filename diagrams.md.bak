# Cheating Detection System – Data Flow Diagram (Level 1)

This document provides a Level 1 Data Flow Diagram (DFD) narrative for the real-time, web-based proctoring system implemented with Flask (backend web server) and Firebase (data storage and authentication support). Level 1 is appropriate here because it decomposes the overall system into major processes while remaining readable for stakeholders; lower levels (Level 2+) can be developed later if finer-grained decomposition is required.

## External Entities
- **Student** – Authenticates, joins sections, takes assigned exams, streams webcam/microphone data, and receives violation warnings or termination notices.
- **Proctor** – Oversees assigned sections, reviews active/finished sessions, and inspects violations and recordings.
- **Administrator** – Manages users, classes/sections, exams, and global system settings.

## Main Processes
1. **User Registration and Login** – Handles creation/authentication of users (students, proctors, administrators) and associates them with classes/sections. Uses Firebase credentials and stores user profiles in Firestore.
2. **Exam Management and Assignment** – Administrators/proctors create exams, assign them to classes/sections, and define availability windows. Exam metadata (questions, timing, class bindings) is stored in Firestore.
3. **Proctored Exam Handling** – Flask serves the exam UI. Students start sessions; the browser captures video/audio and UI events. Session metadata (start/end, class, exam ID) is recorded in Firestore.
4. **Violation Detection and Logging** – Runs in real time during the exam. Client-side detection (tab switch, fullscreen exit, right-click, loud audio, phone via COCO-SSD) and periodic server-side detection (MediaPipe, YOLO) produce violations. Violations are written to Firestore under the corresponding proctoring session.
5. **Report Viewing and Generation** – Proctors/administrators retrieve session data, violations, and recordings from Firestore and storage to review integrity and outcomes.

## Data Stores
- **User Data** – Firestore documents for user profiles, roles, class/section membership, and authentication links.
- **Exam Data** – Firestore documents for exam definitions, availability windows, class/section bindings, and questions.
- **Session Data** – Firestore documents for proctoring sessions (start/end, status, flags/violations, optional video/clip paths).
- **Violation Data** – Stored within session documents as structured flags (type, timestamp, message, evidence links).
- **Result Data** – Firestore documents for exam results (answers, scores, termination flags, timestamps).

## Data Flow (Level 1 Narrative)
- **Student → User Registration and Login → User Data**  
  Students authenticate; their profiles and class/section memberships are stored/updated in User Data.

- **Administrator/Proctor → Exam Management and Assignment → Exam Data**  
  Exams are authored and bound to classes/sections; definitions and schedules are persisted in Exam Data.

- **Exam Management and Assignment → Student**  
  Assigned exams and availability windows are presented to eligible students based on their class/section.

- **Student → Proctored Exam Handling → Session Data**  
  On exam start, a proctoring session document is created/updated with session identifiers, timestamps, and status.

- **Student (browser capture) → Violation Detection and Logging → Violation Data / Session Data**  
  Real-time client-side checks (tab switch, fullscreen exit, right-click, audio loudness, phone via COCO-SSD) and periodic server-side analysis (MediaPipe face/pose, YOLO objects) produce violations that are appended to the session’s Violation Data within Session Data.

- **Violation Detection and Logging → Student**  
  Immediate feedback is provided (warnings, 5-second pauses, or termination at configured thresholds).

- **Proctored Exam Handling → Result Data**  
  Upon submission or forced termination, answers and status (including forced zero score) are written to Result Data; the session status is updated in Session Data.

- **Report Viewing and Generation → Session Data / Violation Data / Result Data**  
  Proctors/administrators retrieve sessions, violations, and results for review; optional recording references (if stored) are read for evidence.

## Notes on Diagram Level
- **Level 1 is appropriate** to convey the principal processes, data stores, and flows without overwhelming detail.  
- Further decomposition (Level 2+) can be produced for specific processes (e.g., “Violation Detection and Logging”) if required for implementation or audit. 


---

# Cheating Detection System – Entity Relationship Diagram (ERD)

This section describes the conceptual Entity Relationship Diagram (ERD) of the Cheating Detection System for Online Examinations. Although the system uses Firebase Firestore, which is a document-oriented NoSQL database rather than a traditional relational DBMS, the entities and relationships can still be modelled conceptually in relational terms. In Firestore, primary keys correspond to document identifiers, and foreign keys are implemented as stored references (IDs) between collections.

## Core Entities and Their Purpose

### 1. `USERS`
- **Purpose**: Represents all system actors, including students, proctors, and administrators.
- **Key attributes** (conceptual):
  - `user_id` (PK, Firestore document ID)
  - `email`, `password` (or external auth UID reference)
  - `name`
  - `role` (e.g. `student`, `proctor`, `admin`)
  - `class_id` (FK to `CLASSES`, indicating the class/section for students)
  - `exams_taken` (array of `exam_id` values, used to prevent retakes)

The `USERS` entity is central for authentication, authorization, and for linking students and proctors to classes and exam participation.

### 2. `CLASSES` (or Sections)
- **Purpose**: Encapsulates course/section information, used to group students and to scope exams to specific cohorts under a proctor.
- **Key attributes**:
  - `class_id` (PK)
  - `course`, `section`
  - `proctor_id` (FK to `USERS.user_id` for the responsible proctor)
  - `join_code` (code students use to self-enrol into the class)

`CLASSES` provides the structural grouping needed so that exams can be assigned to sections and proctors can see only their own classes.

### 3. `EXAMS`
- **Purpose**: Stores exam definitions and their association with classes/sections.
- **Key attributes**:
  - `exam_id` (PK)
  - `title`, `subject`
  - `duration`
  - `start_date`, `end_date` (availability window)
  - `class_id` (FK to `CLASSES.class_id`)
  - `questions` (embedded array of question objects, including correct answers)

`EXAMS` is related to `CLASSES` so that a given exam is visible only to students enrolled in the corresponding section.

### 4. `PROCTORING_SESSIONS`
- **Purpose**: Represents a monitored exam attempt by a specific student for a specific exam.
- **Key attributes**:
  - `session_id` (PK)
  - `student_id` (FK to `USERS.user_id` with role `student`)
  - `exam_id` (FK to `EXAMS.exam_id`)
  - `start_time`, `end_time`
  - `status` (e.g. `active`, `completed`, `terminated`)
  - `flags` (embedded `VIOLATIONS` per type, e.g. `twofaces`, `tab_switch`, `phone`)
  - `video_path`, `clip_path` (optional URLs to stored recordings)

`PROCTORING_SESSIONS` provides the temporal and contextual container within which all real-time cheating detection and logging occurs.

### 5. `VIOLATIONS` (Conceptual, Embedded in `PROCTORING_SESSIONS`)
- **Purpose**: Captures detailed records of suspected cheating or suspicious behaviour during a session.
- **Implementation note**: In Firestore, violations are embedded as arrays within `PROCTORING_SESSIONS.flags` rather than stored in a separate top-level collection. Conceptually, however, they constitute a distinct entity type.
- **Key attributes**:
  - `violation_id` (implicit, often omitted in embedded arrays)
  - `session_id` (FK to `PROCTORING_SESSIONS.session_id`, via parent document)
  - `type` (e.g. `twofaces`, `loud_sound`, `fullscreen_exit`, `tab_switch`, `phone`)
  - `timestamp`
  - `message` (human-readable description)
  - Optional metadata (e.g. evidence link, severity)

The `VIOLATIONS` entity enables fine-grained analysis of behaviour within a session and supports both automated responses (e.g. termination after N strikes) and post-hoc review.

### 6. `RESULTS` (or `EXAM_RESULTS`)
- **Purpose**: Stores the outcome of each student’s attempt at an exam.
- **Key attributes**:
  - `result_id` (PK)
  - `user_id` (FK to `USERS.user_id`)
  - `exam_id` (FK to `EXAMS.exam_id`)
  - `answers` (array aligned with `EXAMS.questions`)
  - `timestamp`
  - `terminated` (boolean; whether the exam was force-ended)
  - `forced_score` (numeric; e.g. 0 when terminated for cheating)

`RESULTS` decouples exam outcome data from the proctoring session, allowing academic scoring and integrity status to be analysed together.

### 7. `SHARED_REPORTS`
- **Purpose**: Provides shareable, token-based links to specific reports for external review (e.g. appeals, external auditors).
- **Key attributes**:
  - `share_id` (PK)
  - `token` (random, URL-safe token)
  - `student_id` (FK to `USERS.user_id`)
  - `exam_id` (FK to `EXAMS.exam_id`)
  - `session_id` (FK to `PROCTORING_SESSIONS.session_id`)
  - `created_at`, `expires_at` (optional expiry)

`SHARED_REPORTS` enables controlled, revocable access to sensitive proctoring data without requiring full system credentials.

## Primary and Foreign Key Relationships (Conceptual)

Although Firestore is schemaless, the system adheres to the following conceptual key relationships:

- **USERS–CLASSES**  
  - `USERS.class_id` → `CLASSES.class_id` (many students per class; each student belongs to at most one primary class in this model).

- **CLASSES–EXAMS**  
  - `EXAMS.class_id` → `CLASSES.class_id` (one class has many exams; each exam is tied to a single class/section).

- **USERS–PROCTORING_SESSIONS–EXAMS**  
  - `PROCTORING_SESSIONS.student_id` → `USERS.user_id`  
  - `PROCTORING_SESSIONS.exam_id` → `EXAMS.exam_id`  
  (each session corresponds to exactly one student and one exam; a student can have multiple sessions across different exams or attempts).

- **PROCTORING_SESSIONS–VIOLATIONS**  
  - `VIOLATIONS.session_id` → `PROCTORING_SESSIONS.session_id` (implemented by embedding violations under `flags` in the session document; one session to many violation records).

- **USERS–RESULTS–EXAMS**  
  - `RESULTS.user_id` → `USERS.user_id`  
  - `RESULTS.exam_id` → `EXAMS.exam_id`  
  (one result per unique `(user_id, exam_id)` in the standard workflow; additional attempts could be modelled as multiple results).

- **SHARED_REPORTS–USERS–EXAMS–PROCTORING_SESSIONS**  
  - `SHARED_REPORTS.student_id` → `USERS.user_id`  
  - `SHARED_REPORTS.exam_id` → `EXAMS.exam_id`  
  - `SHARED_REPORTS.session_id` → `PROCTORING_SESSIONS.session_id`  
  (each shared report is a projection over an existing result and session for a specific student/exam).

## How the ERD Supports System Functionality

### Exam Monitoring
- The linkage `USERS` → `CLASSES` → `EXAMS` ensures that each student only sees exams assigned to their section.
- When a student starts an exam, a `PROCTORING_SESSIONS` document is created with FKs to `USERS` and `EXAMS`. This allows live monitoring and historical analysis of attempts.

### Violation Logging
- During a session, client-side and server-side detectors append structured `VIOLATIONS` into the `flags` field of the corresponding `PROCTORING_SESSIONS` document.
- Because each violation instance records its `type` and `timestamp` and is implicitly linked to a single `session_id`, it is straightforward to:
  - compute per-type violation counts,  
  - enforce termination thresholds (e.g. third violation or immediate phone detection), and  
  - present chronological evidence to proctors and administrators.

### Result Storage and Attempt Locking
- After submission or forced termination, a `RESULTS` document is created with references to `USERS` and `EXAMS`, as well as flags (`terminated`, `forced_score`).  
- The `USERS.exams_taken` array, together with `RESULTS`, allows the system to prevent further attempts on the same exam and to display accurate completion and score data on the student dashboard.

### Report Generation and Sharing
- Proctors and administrators generate reports by querying `PROCTORING_SESSIONS` (for timing, flags, recordings) joined conceptually with `RESULTS` (for scores and termination status) and `USERS`/`EXAMS` (for identity and exam metadata).
- `SHARED_REPORTS` introduces a token-based indirection layer that references `student_id`, `exam_id`, and `session_id`. This supports:
  - public but unguessable report URLs,  
  - time-limited or revocable access via `expires_at`, and  
  - clear traceability from shared artefacts back to core entities.

## Alignment with Firestore

In Firestore, each of the above entities corresponds to a top-level collection (`users`, `classes`, `exams`, `proctoring_sessions`, `exam_results`, `shared_reports`), with violations embedded in `proctoring_sessions.flags`. Primary keys are Firestore document IDs; foreign keys are simple string fields storing related document IDs. This design:

- preserves referential integrity at the application layer,  
- leverages Firestore’s strengths in hierarchical, document-based storage, and  
- remains conceptually compatible with a classical ERD, facilitating reasoning, documentation, and potential future migration to a relational store if required. 


---

# Cheating Detection System – Use Case Diagram (Narrative)

This section provides a textual explanation of the Use Case Diagram for the online examination proctoring system, suitable for inclusion in Chapter 3 of a BSIT capstone manuscript. It identifies the primary actors, their associated use cases, and clarifies the role of automated detection as an assistive—not substitutive—mechanism for proctorial judgement.

## Actors
- **Student** – The examinee who authenticates, joins a section, and attempts assigned exams under proctoring conditions.
- **Proctor** – The invigilator who oversees exams for assigned sections, reviews live/recorded sessions, and evaluates logged violations.
- **Administrator** – The system manager who configures users, classes/sections, and exams, and oversees overall system operations.

## Main Use Cases by Actor

### Student
1. **Register / Login** – Authenticate to the system and establish role-based access (student).
2. **Join Section** – Enrol in a class/section (via join code) to become eligible for assigned exams.
3. **View Available Exams** – See exams scheduled for the student’s section, including availability windows.
4. **Take Proctored Exam** – Launch the exam UI, provide responses, and remain compliant with proctoring policies.
5. **Receive Violation Feedback** – Observe real-time warnings, pauses, or terminations when suspicious behaviour is detected.
6. **View Exam Results** – Review scores and termination status after submission or forced termination.

### Proctor
1. **Login** – Authenticate to gain proctor privileges.
2. **View Assigned Exams and Sections** – Access exams tied to their classes/sections.
3. **Monitor Proctored Sessions** – Observe active sessions (live data, flags, optional recordings).
4. **Review Violations** – Inspect logged suspicious events (timestamps, types, counts) for each session.
5. **Generate / View Reports** – Access summaries of sessions, violations, and outcomes; optionally share via controlled links.

### Administrator
1. **Login** – Authenticate with administrative privileges.
2. **Manage Users** – Create/edit/remove user accounts; assign roles and sections.
3. **Manage Classes/Sections** – Define classes, assign proctors, and generate join codes for student enrolment.
4. **Manage Exams** – Create/edit exams, set availability windows, and bind exams to classes/sections.
5. **Access System Reports** – View system-wide analytics on usage, sessions, and violations.

## Included Use Cases (Cross-Cutting)

- **Detect Suspicious Behaviors** (included within “Take Proctored Exam” and “Monitor Proctored Sessions”)  
  Automated checks (tab switching beyond grace period, fullscreen exit, right-click, keyboard shortcuts, loud audio, multiple faces, phone detection) run in real time. These detections assist proctors by surfacing potential integrity issues.

- **Log Violations** (included within “Detect Suspicious Behaviors”)  
  Each detected event is recorded with type, timestamp, and optional evidence. Violations accumulate per session, enabling threshold-based actions (warnings, pauses, termination) and later review.

- **Generate / View Reports** (used by Proctor and Administrator)  
  Aggregates session data, violations, and results for decision-making, auditing, and potential sharing via tokenized links.

## Human Oversight and Decision-Making
Automated detection is **assistive**: it flags potential misconduct but does **not** replace human judgement. Proctors and administrators remain responsible for interpreting violations, determining severity, and making final decisions (e.g., upholding a termination, allowing a retake, or escalating for review). The system’s logging and reporting functions are designed to support transparent, evidence-based human oversight.


---

# Cheating Detection System – System Architecture Diagram (Narrative)

This section describes the system architecture for the online proctoring solution as a client–server model with cloud-hosted data services. The description is aligned with the implemented stack: a PWA frontend, a Flask backend (REST API), and Firebase for authentication and data storage. Communication occurs over HTTPS, and browser-based AI modules perform lightweight detection to reduce server load. Claims are scoped to the implemented capabilities.

## Client-Side Components (Frontend)
- **PWA Frontend** (HTML/JS/CSS served by Flask): Renders the exam UI, dashboards, and management interfaces. Runs entirely in the browser and can operate with limited offline resilience for UI (not for proctoring).
- **In-Browser Detection Modules** (assistive, not authoritative):
  - MediaPipe Face Mesh (browser): face/landmark detection and basic head pose cues.
  - COCO-SSD (browser): object detection for phones and similar items.
  - Event Monitors: tab visibility, fullscreen state, right-click, keyboard shortcuts, window resize, audio loudness (Web Audio API).
  - Violation Feedback: displays warnings, pauses, or termination notices according to configured thresholds.
  
These browser modules reduce round-trips and offload lightweight detection from the server. They flag potential issues but do not replace human or server-side validation.

## Backend Components (Flask Server / REST API)
- **Flask Application**: Hosts the web pages, exposes REST endpoints for:
  - `/predict` (optional server-side frame analysis: YOLO/MediaPipe, throttled to low frequency)
  - `/frontend_alert` (logs browser-detected violations)
  - Session lifecycle (`/start_session`, `/end_session`, `/get_session_status`)
  - Data CRUD (`/add`, `/select`, `/edit/<id>`, `/delete/<id>`)
  - Authentication helpers (`/login`, `/logout`)
  - Video upload (`/upload_video`)
- **Server-Side Detection (Optional/Throttled)**: When enabled, processes frames sent from the client (low FPS) using OpenCV/MediaPipe/YOLO. This complements—but does not replace—browser detection.

## Firebase Integration
- **Authentication**: Firebase Auth (or Admin SDK linkage) to validate identities; user roles stored in Firestore documents.
- **Data Storage (Firestore)**: Collections for users, classes/sections, exams, proctoring_sessions, exam_results, shared_reports. Violations are stored as flags within proctoring_sessions.
- **(Optional) Storage**: If configured, recordings or clips can be stored in cloud storage; otherwise, local storage is ephemeral on free-tier hosts.

## Communication Flow (HTTPS)
1. **Page and Asset Delivery**: Browser requests pages and static assets from Flask over HTTPS.
2. **Session Setup**: Student starts an exam; client calls `/start_session` to create a proctoring session in Firestore.
3. **Real-Time Client Detection**: Browser modules run locally and send alerts via `/frontend_alert` when suspicious events occur.
4. **Optional Server Detection**: At a throttled interval, the browser may POST a frame to `/predict` for heavier checks. Responses return processed overlays and detection metadata.
5. **Violation Logging**: Both client- and server-triggered flags are written to Firestore under `proctoring_sessions.flags`.
6. **Result Submission**: On submit or forced termination, the client finalizes results via `exam_results` and updates session status.
7. **Reporting**: Proctors/admins fetch sessions, violations, and results via REST calls; shared reports may be served via tokenized links.

## Role of Browser-Based AI Modules
- **Assistive, Not Authoritative**: Client-side models (MediaPipe Face Mesh, COCO-SSD) provide quick signals (e.g., multiple faces, phone detection). They reduce server load and latency but do not constitute final adjudication.
- **Human Oversight**: Proctors and administrators review logged violations and session evidence to make final decisions. Automated actions (pauses/termination) are configurable safeguards, not substitutes for human judgement.

## Scope and Capability Note
- The architecture reflects a practical client–server design with cloud-hosted data services. It avoids overclaiming: browser AI is lightweight and assistive; server-side detection is optional and throttled; human review remains central for exam integrity decisions. 
