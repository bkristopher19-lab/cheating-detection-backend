# Diagrams Implementation Guide (draw.io)

This guide explains how to implement the diagrams for the **Cheating Detection System for Online Examinations** using **draw.io** (now diagrams.net). It focuses on four diagrams typically required in a BSIT capstone Chapter 3:

- Data Flow Diagram (Level 1)
- Entity Relationship Diagram (ERD)
- Use Case Diagram
- System Architecture Diagram

The textual specifications for each diagram are in `diagrams.md`. This document translates those specifications into concrete drawing steps.

---

## 1. Data Flow Diagram (Level 1)

**Goal:** Show how data moves between external entities, main processes, and data stores for the cheating detection system.

### 1.1. Setup in draw.io
1. Open `draw.io` (diagrams.net) and create a new **Blank Diagram**.
2. Set the page name to **DFD_Level1**.
3. From the left panel, enable the **UML** or **Flowchart** shape set (View → Shapes).

### 1.2. Elements to Place
Use standard DFD notation (or clear equivalents):
- **External Entities**: Rounded rectangles or “Entity” shapes.
  - `Student`
  - `Proctor`
  - `Administrator`
- **Processes**: Rounded rectangles or circles labelled with process names:
  1. `User Registration and Login`
  2. `Exam Management and Assignment`
  3. `Proctored Exam Handling`
  4. `Violation Detection and Logging`
  5. `Report Viewing and Generation`
- **Data Stores**: Open-ended rectangles or “Database” symbols:
  - `User Data`
  - `Exam Data`
  - `Session Data`
  - `Violation Data`
  - `Result Data`

### 1.3. Data Flows
Use **arrows** to connect entities → processes → data stores, based on the narrative in `diagrams.md`:
- `Student → User Registration and Login → User Data`
- `Administrator/Proctor → Exam Management and Assignment → Exam Data`
- `Exam Management and Assignment → Student` (available exams)
- `Student → Proctored Exam Handling → Session Data`
- `Proctored Exam Handling ↔ Violation Detection and Logging → Violation Data / Session Data`
- `Proctored Exam Handling → Result Data`
- `Proctor/Administrator → Report Viewing and Generation ↔ Session Data / Violation Data / Result Data`

Add short labels to arrows where helpful (e.g., “credentials”, “exam metadata”, “flags”, “results”).

---

## 2. Entity Relationship Diagram (ERD)

**Goal:** Model the main data entities and relationships, conceptually aligned with Firestore collections.

### 2.1. Setup in draw.io
1. Create a new page named **ERD**.
2. Enable the **Entity Relation** or **Database** shapes.

### 2.2. Entities (Tables)
Draw rectangles with a divided header/body to represent entities. Suggested entities:
- `USERS`
- `CLASSES`
- `EXAMS`
- `PROCTORING_SESSIONS`
- `RESULTS` (or `EXAM_RESULTS`)
- `SHARED_REPORTS`
- (Optionally) `VIOLATIONS` as a conceptual entity, but note it is embedded in `PROCTORING_SESSIONS.flags`.

Inside each entity, list key attributes (PK underlined or marked with “(PK)”, FKs with “(FK)”):
- `USERS(user_id PK, email, name, role, class_id FK, exams_taken[])`
- `CLASSES(class_id PK, course, section, proctor_id FK, join_code)`
- `EXAMS(exam_id PK, title, subject, duration, start_date, end_date, class_id FK, questions[])`
- `PROCTORING_SESSIONS(session_id PK, student_id FK, exam_id FK, start_time, end_time, status, flags, video_path, clip_path)`
- `RESULTS(result_id PK, user_id FK, exam_id FK, answers[], timestamp, terminated, forced_score)`
- `SHARED_REPORTS(share_id PK, token, student_id FK, exam_id FK, session_id FK, created_at, expires_at)`

### 2.3. Relationships
Use crow’s foot or simple line notation:
- `USERS (1) — (M) PROCTORING_SESSIONS` via `student_id`.
- `EXAMS (1) — (M) PROCTORING_SESSIONS` via `exam_id`.
- `CLASSES (1) — (M) EXAMS` via `class_id`.
- `CLASSES (1) — (M) USERS` via `class_id` (for students).
- `USERS (1) — (M) CLASSES` via `proctor_id` (for proctors).
- `USERS (1) — (M) RESULTS` via `user_id`.
- `EXAMS (1) — (M) RESULTS` via `exam_id`.
- `PROCTORING_SESSIONS (1) — (M) VIOLATIONS` (conceptual; embedded).
- `SHARED_REPORTS` linked to `USERS`, `EXAMS`, `PROCTORING_SESSIONS` via FK lines.

Label relationships where needed (e.g., “takes”, “monitors”, “belongs to class”).

---

## 3. Use Case Diagram

**Goal:** Show how each actor interacts with the system’s main functions.

### 3.1. Setup in draw.io
1. Create a new page named **UseCase**.
2. Enable the **UML** shape library.

### 3.2. Actors
Place UML stick-figure icons on the left/right:
- `Student`
- `Proctor`
- `Administrator`

### 3.3. Use Cases
Use UML ovals to represent use cases. Group them near the relevant actor:

**Student:**
- `Register / Login`
- `Join Section`
- `View Available Exams`
- `Take Proctored Exam`
- `Receive Violation Feedback`
- `View Exam Results`

**Proctor:**
- `Login`
- `View Assigned Exams and Sections`
- `Monitor Proctored Sessions`
- `Review Violations`
- `Generate / View Reports`

**Administrator:**
- `Login`
- `Manage Users`
- `Manage Classes/Sections`
- `Manage Exams`
- `Access System Reports`

### 3.4. Include / Extend Relationships
- Draw dashed arrows with «include» from:
  - `Take Proctored Exam` → `Detect Suspicious Behaviors`
  - `Detect Suspicious Behaviors` → `Log Violations`
- Optionally, «include» from `Generate / View Reports` to `Log Violations` and `Access Session Data`.

Connect each actor to its use cases with solid lines. Place a note near `Detect Suspicious Behaviors` stating: *“Automated detection assists human proctors; it does not replace human decision-making.”*

---

## 4. System Architecture Diagram

**Goal:** Illustrate the high-level client–server architecture with Firebase integration and browser-based AI modules.

### 4.1. Setup in draw.io
1. Create a new page named **Architecture**.
2. Use simple rectangles, cloud icons, and arrows.

### 4.2. Components to Draw

**Left (Client Side – Browser/PWA):**
- Box labelled `PWA Frontend (Browser)` containing:
  - `Exam UI & Dashboards`
  - `In-Browser Detection` (MediaPipe Face Mesh, COCO-SSD, Event Monitors)
  - `Violation Feedback (Warnings, Pauses, Termination)`

**Middle (Backend – Flask Server):**
- Box labelled `Flask Server (REST API)` with sublabels:
  - `Routing & Templates`
  - `Endpoints: /predict, /frontend_alert, /start_session, /end_session, /select, /edit, /add, /delete, /upload_video`
  - `Optional Server-Side Detection (OpenCV, MediaPipe, YOLO)`

**Right (Cloud Services – Firebase):**
- Cloud icon or box labelled `Firebase`:
  - `Firestore (users, classes, exams, proctoring_sessions, exam_results, shared_reports)`
  - `Auth (user identity/roles)`
  - *(Optional)* `Storage (recordings/clips)`

### 4.3. Connections
- Arrow from **PWA Frontend** → **Flask Server** labelled `HTTPS (REST requests)`.
- Arrow from **Flask Server** → **Firebase** labelled `Firestore / Auth operations`.
- Optional arrow from **PWA Frontend** → **Firebase** if using front-channel Auth.
- Annotate near the client box: *“Browser-based AI modules are assistive; they flag suspicious behaviour but do not replace human judgement.”*

---

## 5. Exporting and Using the Diagrams

1. For each page (**DFD_Level1**, **ERD**, **UseCase**, **Architecture**), export to:
   - **PNG** or **JPG** for inclusion in the manuscript.
   - **PDF** for printing or appendices.
2. Name files clearly, e.g.:
   - `dfd_level1_cheating_detection.png`
   - `erd_cheating_detection.png`
   - `usecase_cheating_detection.png`
   - `architecture_cheating_detection.png`
3. Reference the diagrams in Chapter 3 (e.g., “Figure 3.x: System Architecture Diagram of the Cheating Detection System”).

These steps ensure that the diagrams in the manuscript directly reflect the system described in `README.md` and `diagrams.md`, and that they can be reproduced or edited easily in draw.io. 





