# TODO: Implement Missing Functionalities in detection.py

## Investigation Findings:
- YOLO model loaded but not used in /predict route
- extract_keypoints function defined but never called
- Frontend references phone detection, audio monitoring, clip recording not implemented in backend
- Firebase initialization duplicated at end of file

- No backend support for clip generation and upload
- Session management routes missing

## Plan:
- [x] Add object detection (phone, etc.) using YOLO in /predict route

- [ ] Add backend support for clip generation and upload
- [ ] Utilize extract_keypoints for potential ML model integration
- [ ] Fix Firebase initialization duplication
- [ ] Add session management routes
- [x] Remove "not looking", "multiple voices", "copy attempt", "dev tools", "noface", "paste attempt", "proctoring toggle", "text selection" detections

## Dependent Files to be edited:
- detection.py (main backend file)

## Followup steps:
- [ ] Install required audio processing dependencies (librosa, pydub)
- [ ] Test detection features after implementation
- [ ] Verify Firebase updates and video handling
- [ ] Check console logs for errors during proctoring

## Implementation Steps:
- [ ] Step 1: Install audio processing dependencies (librosa, pydub)
- [ ] Step 2: Implement audio detection routes (/detect_audio)
- [ ] Step 3: Add clip generation support in /predict route
- [ ] Step 4: Integrate extract_keypoints in /predict route
- [ ] Step 5: Fix Firebase initialization duplication
- [ ] Step 6: Add session management routes (/start_session, /end_session, /get_session_status)
- [x] Step 8: Implement automatic deletion of reports after 14 days
- [ ] Step 7: Test all implemented features
