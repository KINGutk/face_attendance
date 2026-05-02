"""
Face Attendance System - Startup Script
========================================
Run this from the project root to start the Flask app.

Usage:
    python run.py
"""
import os
import sys

# Fix Windows console encoding for emoji characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Set the working directory to face_attendance_web BEFORE importing app
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_attendance_web')
os.chdir(app_dir)
sys.path.insert(0, app_dir)

from app import app  # type: ignore[import-not-found]  # noqa: E402 — path set dynamically above

if __name__ == '__main__':
    # Use use_reloader=False to avoid the cwd/reloader conflict on Windows
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true',
        use_reloader=False
    )
