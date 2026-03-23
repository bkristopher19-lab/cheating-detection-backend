"""
Data validation utilities for the proctoring system.
Provides server-side validation for API inputs to prevent invalid data, injection, and XSS.
"""
import re
from typing import Optional, Tuple

# Allowed Firestore collection names (whitelist to prevent arbitrary collection access)
ALLOWED_COLLECTIONS = frozenset({
    'users', 'exams', 'classes', 'proctoring_sessions', 'shared_reports', 'exam_results'
})

# Allowed frontend alert types
ALLOWED_ALERT_TYPES = frozenset({
    'keyboardShortcut', 'tabSwitch', 'rightClick', 'fullscreenExit', 'captions'
})

# Allowed manual violation types (for proctor tagging while reviewing video)
MANUAL_VIOLATION_TYPES = frozenset({
    'twofaces', 'multiplefaces', 'loud_sound', 'phone', 'object_detected_cell_phone',
    'fullscreen_exit', 'keyboard_shortcut', 'right_click', 'tab_switch',
    'face_absent', 'not_looking', 'captions', 'other'
})

# Allowed video file extensions for uploads
ALLOWED_VIDEO_EXTENSIONS = frozenset({'.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v'})

# Firestore document IDs: alphanumeric, underscores, hyphens (Firebase default format)
# Max 1500 chars (Firestore limit), but we use a reasonable max for validation
ID_MAX_LENGTH = 128
NAME_MIN_LENGTH = 2
NAME_MAX_LENGTH = 100
PASSWORD_MIN_LENGTH = 6
PASSWORD_MAX_LENGTH = 128
EMAIL_MAX_LENGTH = 254
OTP_LENGTH = 6
EXAM_DURATION_MIN = 5
EXAM_DURATION_MAX = 120
ALLOWED_EMAIL_DOMAINS = frozenset({'gmail.com', 'yahoo.com'})

# Simple email regex (covers most valid emails; not RFC 5322 exhaustive)
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)
# Firestore ID: alphanumeric, underscore, hyphen
ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
# Name: letters, spaces, hyphens, apostrophes (no HTML/script)
NAME_PATTERN = re.compile(r'^[\w\s\-. \']+$', re.UNICODE)
# OTP: exactly 6 digits
OTP_PATTERN = re.compile(r'^\d{6}$')


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """Validate email format + allowlisted domains. Returns (valid, error_message)."""
    if not email or not isinstance(email, str):
        return False, 'Email is required'
    email = email.strip().lower()
    if len(email) > EMAIL_MAX_LENGTH:
        return False, 'Email is too long'
    if not EMAIL_PATTERN.match(email):
        return False, 'Invalid email format'
    domain = email.rsplit('@', 1)[-1]
    if domain not in ALLOWED_EMAIL_DOMAINS:
        return False, 'Only Gmail and Yahoo email addresses are allowed'
    return True, None


def validate_otp(otp: str) -> Tuple[bool, Optional[str]]:
    """Validate OTP: exactly 6 digits."""
    if not otp or not isinstance(otp, str):
        return False, 'OTP is required'
    otp = otp.strip()
    if not OTP_PATTERN.match(otp):
        return False, 'OTP must be exactly 6 digits'
    return True, None


def validate_name(name: str) -> Tuple[bool, Optional[str]]:
    """Validate display name: 2-100 chars, safe characters."""
    if not name or not isinstance(name, str):
        return False, 'Name is required'
    name = name.strip()
    if len(name) < NAME_MIN_LENGTH:
        return False, f'Name must be at least {NAME_MIN_LENGTH} characters'
    if len(name) > NAME_MAX_LENGTH:
        return False, f'Name must be at most {NAME_MAX_LENGTH} characters'
    # Disallow potential XSS/HTML
    if '<' in name or '>' in name or '&' in name or ';' in name:
        return False, 'Name contains invalid characters'
    if not NAME_PATTERN.match(name):
        return False, 'Name may only contain letters, numbers, spaces, hyphens, and apostrophes'
    return True, None


def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """Validate password length (Firebase requires min 6)."""
    if not password or not isinstance(password, str):
        return False, 'Password is required'
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f'Password must be at least {PASSWORD_MIN_LENGTH} characters'
    if len(password) > PASSWORD_MAX_LENGTH:
        return False, f'Password must be at most {PASSWORD_MAX_LENGTH} characters'
    return True, None


def validate_collection(collection: str) -> Tuple[bool, Optional[str]]:
    """Validate collection name is in whitelist."""
    if not collection or not isinstance(collection, str):
        return False, 'Collection is required'
    if collection not in ALLOWED_COLLECTIONS:
        return False, f'Invalid collection: {collection}'
    return True, None


def validate_firestore_id(id_str: str, field_name: str = 'id') -> Tuple[bool, Optional[str]]:
    """Validate Firestore document ID format."""
    if not id_str or not isinstance(id_str, str):
        return False, f'{field_name} is required'
    id_str = id_str.strip()
    if len(id_str) > ID_MAX_LENGTH:
        return False, f'{field_name} is too long'
    if not ID_PATTERN.match(id_str):
        return False, f'{field_name} contains invalid characters'
    return True, None


def validate_alert_type(alert_type: str) -> Tuple[bool, Optional[str]]:
    """Validate frontend alert type is allowed."""
    if not alert_type or not isinstance(alert_type, str):
        return False, 'Alert type is required'
    if alert_type not in ALLOWED_ALERT_TYPES:
        return False, f'Unknown alert type: {alert_type}'
    return True, None


def validate_manual_violation_type(violation_type: str) -> Tuple[bool, Optional[str]]:
    """Validate manual violation type for proctor tagging."""
    if not violation_type or not isinstance(violation_type, str):
        return False, 'Violation type is required'
    if violation_type not in MANUAL_VIOLATION_TYPES:
        return False, f'Unknown violation type: {violation_type}'
    return True, None


def validate_video_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """Validate uploaded file has allowed video extension."""
    if not filename or not isinstance(filename, str):
        return False, 'Filename is required'
    lower = filename.lower()
    ext = '.' + lower.rsplit('.', 1)[-1] if '.' in lower else ''
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f'Invalid video format. Allowed: {", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))}'
    return True, None


def validate_user_data(data: dict) -> Tuple[bool, Optional[str]]:
    """Validate user document data for /add?collection=users."""
    if not data or not isinstance(data, dict):
        return False, 'Invalid data: must be a JSON object'
    # Require basic fields
    name = data.get('name') or data.get('displayName')
    if not name:
        return False, 'User data must include name or displayName'
    valid, err = validate_name(str(name))
    if not valid:
        return False, err
    email = data.get('email')
    if email:
        valid, err = validate_email(str(email))
        if not valid:
            return False, err
    # Sanitize: don't allow arbitrary keys that could override system fields
    # (optional: whitelist keys; for now we allow common user fields)
    return True, None


def validate_exam_data(data: dict, require_all_fields: bool = True) -> Tuple[bool, Optional[str]]:
    """Validate exam document data. Enforces duration range of 5-120 minutes."""
    if not data or not isinstance(data, dict):
        return False, 'Invalid data: must be a JSON object'

    if require_all_fields:
        for field in ('title', 'subject', 'exam_type', 'class_id', 'questions', 'duration'):
            if field not in data:
                return False, f'Exam data must include {field}'

    if 'duration' in data:
        raw_duration = data.get('duration')
        try:
            duration = int(raw_duration)
        except (TypeError, ValueError):
            return False, 'Duration must be a whole number of minutes'
        if duration < EXAM_DURATION_MIN or duration > EXAM_DURATION_MAX:
            return False, f'Duration must be between {EXAM_DURATION_MIN} and {EXAM_DURATION_MAX} minutes'

    if 'questions' in data and data.get('questions') is not None and not isinstance(data.get('questions'), list):
        return False, 'questions must be an array'

    if 'title' in data:
        title = str(data.get('title', '')).strip()
        if not title:
            return False, 'Exam title is required'

    if 'subject' in data:
        subject = str(data.get('subject', '')).strip()
        if not subject:
            return False, 'Exam subject is required'

    return True, None


def validate_session_id(session_id: str) -> Tuple[bool, Optional[str]]:
    """Validate proctoring session ID."""
    return validate_firestore_id(session_id, 'session_id')


def validate_cutoff_days(days: int) -> Tuple[bool, Optional[str]]:
    """Validate cutoff_days for delete_old_reports."""
    if not isinstance(days, int):
        return False, 'cutoff_days must be an integer'
    if days < 1:
        return False, 'cutoff_days must be at least 1'
    if days > 365:
        return False, 'cutoff_days must be at most 365'
    return True, None
