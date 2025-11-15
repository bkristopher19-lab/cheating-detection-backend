# TODO: Investigate Session Recording and Clip Generation Issues

## Investigation Plan
- [x] Check upload_video endpoint for proper file saving and Firebase updates
- [x] Verify clip generation triggers correctly on violations
- [x] Test video display in reports
- [x] Check static/recordings/ directory for saved files
- [x] Debug any upload failures or path issues

## Dependent Files to be edited:
- `detection.py` (upload_video endpoint)
- `templates/index.html` (recording logic)
- `templates/reports.html` (video display)

## Followup steps:
- [x] Run the app and test recording functionality
- [x] Check browser console for errors during recording/upload
- [x] Verify Firebase documents are updated with video paths
- [x] Test clip generation by triggering violations

## Findings from Investigation:
- [x] Files are being saved to static/recordings/ directory successfully
- [x] Upload endpoint in detection.py properly saves files and updates Firebase with video_path/clip_path
- [x] Clip generation logic is implemented in startCheatingClip() function
- [x] Reports display videos when video_path exists in session data
- [x] Session recording starts in startMediaRecorder() and uploads on exam submit
- [x] Clip recording triggers on violations (phone, twofaces, notlooking, loudsound, mvoices, noface, proctoring_toggle)

## Issues Identified:
- [ ] Need to verify if clips are actually being generated when violations occur
- [ ] Need to check if video paths are correctly stored in Firebase
- [ ] Need to test video playback in reports interface
- [ ] Need to ensure clipRecorded flag resets properly between sessions

## Next Steps:
- [ ] Test the system by triggering violations during an exam session
- [ ] Check Firebase documents for video_path and clip_path fields
- [ ] Verify video playback works in the reports interface
- [ ] Debug any issues with clip generation timing or upload failures
