from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session, flash
import mysql.connector
from datetime import datetime, timedelta
import os
import sys
import json
import urllib.request
import base64
import numpy as np
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import threading
import time
import atexit
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler

# Fix Windows console encoding for emoji characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ==================================================
# 🤖 ADVANCED AI IMPORTS (YOLOv8 + PyTorch)
# ==================================================
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# ==================================================
# 🌍 LOAD ENVIRONMENT VARIABLES
# ==================================================
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables.")

# ==================================================
# 🔧 FLASK APP CONFIG
# ==================================================
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super_secure_authentic_key_2026')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for face images

# ==================================================
# 🧠 AI MODEL LOADING (YOLOv8 + ResNet50)
# ==================================================
print("🔄 Loading YOLOv8 and ResNet50 models...")

MODEL_PATH = 'yolov8n-face.pt'
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print("⚠️ Model corrupted. Redownloading safe version...")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    urllib.request.urlretrieve("https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt", MODEL_PATH)
    yolo_model = YOLO(MODEL_PATH)

# PyTorch ResNet50 for 512D Face Maps
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = resnet50(weights='DEFAULT')
embedding_model.fc = torch.nn.Linear(2048, 512)
embedding_model = embedding_model.to(device)
embedding_model.eval()

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_face_embedding(img_bgr):
    """Generates a 512-dimensional math map of the face using PyTorch"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = face_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedding_model(tensor)
    return torch.nn.functional.normalize(emb, dim=1).squeeze().cpu().numpy()


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def decode_b64_image(b64_string):
    """Decode a data-URL / raw base64 string into an OpenCV BGR image."""
    if not b64_string:
        return None
    if "," in b64_string:
        _, encoded = b64_string.split(",", 1)
    else:
        encoded = b64_string
    np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def is_image_blurry(img_bgr, threshold=50):
    """Returns (is_blurry, score) using Laplacian variance."""
    if img_bgr is None:
        return True, 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score


def detect_and_crop_face(img_bgr, conf_threshold=0.55, min_area_ratio=0.08):
    """Runs YOLOv8 face detection. Returns (face_crop, error_message)."""
    if img_bgr is None:
        return None, "Could not decode image"

    results = yolo_model(img_bgr, verbose=False)
    faces = [box for r in results for box in r.boxes if box.conf[0] >= conf_threshold]

    if not faces:
        return None, "⚠️ No face detected. Look at the camera."
    if len(faces) > 1:
        return None, "⚠️ Multiple faces detected. Only one person at a time."

    box = faces[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    face_area = (x2 - x1) * (y2 - y1)
    image_area = img_bgr.shape[0] * img_bgr.shape[1]

    if face_area < (image_area * min_area_ratio):
        return None, "⚠️ Face too small. Please move closer."

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "⚠️ Crop error. Please try again."

    return crop, None


# ==================================================
# 📧 EMAIL CONFIGURATION
# ==================================================
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'khushaldegreecollege@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'ypfb ljkv zfgv hriq')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])
mail = Mail(app)

# ==================================================
# 💾 DATABASE CONNECTION
# ==================================================


def get_db_connection():
    try:
        db_host = os.environ.get('DB_HOST', 'localhost')
        conn_params = {
            'host': db_host,
            'port': int(os.environ.get('DB_PORT', 3306)),
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASS', ''),
            'database': os.environ.get('DB_NAME', 'face_attendance_db'),
        }
        if db_host != 'localhost' and db_host != '127.0.0.1':
            conn_params['ssl_disabled'] = False
        return mysql.connector.connect(**conn_params)
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        return None


# ==================================================
# 🖼️ FACE CACHE (DIRECT FROM DATABASE)
# ==================================================
KNOWN_ENCODINGS = []
KNOWN_NAMES = []
KNOWN_ROLLS = []


def load_known_faces():
    """Loads Face Maps (JSON) directly from TiDB."""
    print("🔄 Loading AI Face Maps from Database...")
    global KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS
    KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS = [], [], []

    db = get_db_connection()
    if not db:
        return

    try:
        cursor = db.cursor(dictionary=True)
        # Ensure face_data column exists
        try:
            cursor.execute("ALTER TABLE students ADD COLUMN face_data LONGTEXT;")
            db.commit()
            print("✅ 'face_data' column added to DB!")
        except Exception:
            pass

        cursor.execute("SELECT roll_no, name, face_data FROM students WHERE status = 'approved'")
        for student in cursor.fetchall():
            if student['face_data']:
                try:
                    encodings_list = json.loads(student['face_data'])
                    for enc in encodings_list:
                        KNOWN_ENCODINGS.append(np.array(enc))
                        KNOWN_NAMES.append(student['name'])
                        KNOWN_ROLLS.append(student['roll_no'])
                except Exception as e:
                    print(f"⚠️ Error parsing JSON for {student['name']}: {e}")
        print(f"✅ Loaded {len(KNOWN_ENCODINGS)} total Math Maps.")
    finally:
        db.close()


load_known_faces()


@app.route('/reload_faces')
def reload_faces():
    load_known_faces()
    return jsonify({"success": True, "message": f"Face DB Reloaded. {len(KNOWN_ENCODINGS)} maps loaded."})


# ==================================================
# 🔑 AUTHENTICATION WRAPPERS
# ==================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def professor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'professor':
            return redirect(url_for('professor_login'))
        return f(*args, **kwargs)
    return decorated


# ==================================================
# 🔄 TOAST NOTIFICATION SYSTEM (for live dashboards)
# ==================================================
last_detection = {}


@app.route('/last_detection')
def get_last_detection():
    return jsonify(last_detection)


@app.route('/clear_detection')
def clear_detection():
    global last_detection
    last_detection = {}
    return jsonify({"status": "cleared"})


def update_detection(name, roll, subject, status, message):
    global last_detection
    last_detection = {
        "name": name,
        "roll": roll,
        "subject": subject,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }


# ==================================================
# 📧 EMAIL NOTIFICATION FUNCTIONS
# ==================================================

def send_attendance_notification(student_email, student_name, status, subject, date, time=None):
    """Send attendance notification to student"""
    try:
        with app.app_context():
            if status == "Present":
                subject_line = f"✅ Khushal College - Present for {subject}"
                body = f"""
Dear {student_name},

Your attendance has been marked as PRESENT at Khushal Degree College:

📚 Subject: {subject}
📅 Date: {date}
⏰ Time: {time if time else 'During class hours'}

Keep up the good attendance! 🎉

Best regards,
Face Attendance System
Khushal Degree College
                """
            elif status == "Absent":
                subject_line = f"⚠️ Khushal College - Absent for {subject}"
                body = f"""
Dear {student_name},

Your attendance has been marked as ABSENT at Khushal Degree College:

📚 Subject: {subject}
📅 Date: {date}

Please contact your professor if this is incorrect.

Best regards,
Face Attendance System
Khushal Degree College
                """
            else:
                return False

            msg = Message(subject=subject_line, recipients=[student_email], body=body)
            mail.send(msg)
            print(f"📧 Attendance notification sent to {student_email}")
            return True
    except Exception as e:
        print(f"❌ Failed to send attendance email to {student_email}: {e}")
        return False


def send_leave_status_notification(student_email, student_name, status, subject, start_date, end_date, purpose=None):
    """Send leave application status notification"""
    try:
        with app.app_context():
            if status == "Approved":
                subject_line = f"✅ Khushal College - Leave Approved for {subject}"
                body = f"""
Dear {student_name},

Your leave application has been APPROVED at Khushal Degree College ✅

📚 Subject: {subject}
🎯 Purpose: {purpose if purpose else 'Leave'}
📅 Period: {start_date} to {end_date}

Your attendance will be marked as "Leave" for this period.

Best regards,
Face Attendance System
Khushal Degree College
                """
            elif status == "Rejected":
                subject_line = f"❌ Khushal College - Leave Rejected for {subject}"
                body = f"""
Dear {student_name},

Your leave application has been REJECTED at Khushal Degree College ❌

📚 Subject: {subject}
🎯 Purpose: {purpose if purpose else 'Leave'}
📅 Period: {start_date} to {end_date}

Please contact college administration if you have questions.

Best regards,
Face Attendance System
Khushal Degree College
                """
            else:
                return False

            msg = Message(subject=subject_line, recipients=[student_email], body=body)
            mail.send(msg)
            print(f"📧 Leave status notification sent to {student_email}")
            return True
    except Exception as e:
        print(f"❌ Failed to send leave status email to {student_email}: {e}")
        return False


def send_attendance_emails_in_background(email_data_list):
    """Send attendance emails in a background thread (non-blocking)."""
    def email_worker():
        print(f"🎯 Background Email Task Started: Sending {len(email_data_list)} emails...")
        success_count = 0
        fail_count = 0

        for i, email_data in enumerate(email_data_list):
            try:
                with app.app_context():
                    result = send_attendance_notification(
                        student_email=email_data['student_email'],
                        student_name=email_data['student_name'],
                        status=email_data['status'],
                        subject=email_data['subject'],
                        date=email_data['date'],
                        time=email_data.get('time')
                    )
                    if result:
                        success_count += 1
                    else:
                        fail_count += 1

                if i < len(email_data_list) - 1:
                    time.sleep(1)
            except Exception as e:
                fail_count += 1
                print(f"💥 Background Email {i+1}/{len(email_data_list)}: Error for {email_data['student_email']} - {e}")

        print(f"🎯 Background Email Task Completed: {success_count} successful, {fail_count} failed")

    thread = threading.Thread(target=email_worker)
    thread.daemon = True
    thread.start()


# ==================================================
# ⚡ INSTANT PHOTO CHECK (YOLOv8)
# ==================================================
@app.route('/check_photo_quality', methods=['POST'])
def check_photo_quality():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"valid": False, "error": "No image data"})

        img = decode_b64_image(image_data)
        if img is None:
            return jsonify({"valid": False, "error": "Could not decode image"})

        # 1. Blur Check
        is_blur, blur_score = is_image_blurry(img, threshold=50)
        if is_blur:
            return jsonify({"valid": False, "error": "⚠️ Too Blurry. Hold steady!"})

        # 2. YOLO Face Detection + Size Check
        crop, err = detect_and_crop_face(img)
        if err:
            return jsonify({"valid": False, "error": err})

        return jsonify({"valid": True})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


# ==================================================
# ⚡ LIVE ATTENDANCE API (Cosine Similarity)
# ==================================================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    db = None
    cursor = None
    try:
        # ⚠️ DB EMPTY FIX: Agar dusre worker ki memory khali ho, to foran load karo
        if not KNOWN_ENCODINGS:
            load_known_faces()

        data = request.json

        image_data = data.get('image')
        if not image_data:
            return jsonify({"message": "No Image", "color": "red", "current_class": "--"})

        img = decode_b64_image(image_data)
        if img is None:
            return jsonify({"message": "Decode error", "color": "red", "current_class": "--"})

        # Get Current Class
        db = get_db_connection()
        if not db:
            return jsonify({"message": "DB connection error", "color": "red", "current_class": "--"})
        cursor = db.cursor(dictionary=True)
        now = datetime.now()
        date_today = now.date()
        time_now = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")

        cursor.execute(
            "SELECT * FROM classes WHERE day_of_week=%s AND start_time<=%s AND end_time>=%s LIMIT 1",
            (day_name, time_now, time_now)
        )
        current_class = cursor.fetchone()
        class_info = f"{current_class['subject_name']} ({current_class['semester']})" if current_class else "No Active Class"

        if not KNOWN_ENCODINGS:
            return jsonify({"message": "DB Empty", "color": "orange", "current_class": class_info})

        # YOLO Detection (downscale for speed)
        small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        results = yolo_model(small, verbose=False)
        faces = [box for r in results for box in r.boxes if box.conf[0] >= 0.5]

        if not faces:
            return jsonify({"message": "No face detected", "color": "orange", "current_class": class_info})

        best_box = max(faces, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = [v * 2 for v in map(int, best_box.xyxy[0])]
        face_crop = img[y1:y2, x1:x2]

        if face_crop.size == 0:
            return jsonify({"message": "Crop error", "color": "red", "current_class": class_info})

        # PyTorch Embedding & Cosine Similarity Match
        query_emb = get_face_embedding(face_crop)
        sims = [cosine_similarity(query_emb, k) for k in KNOWN_ENCODINGS]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        THRESHOLD = 0.72  # Strict accuracy threshold
        if best_sim < THRESHOLD:
            update_detection("Unknown", "Unknown", class_info, "unknown", "⚠️ Unknown Face Detected!")
            return jsonify({"message": "Unknown Face", "color": "red", "current_class": class_info})

        name = KNOWN_NAMES[best_idx]
        roll = KNOWN_ROLLS[best_idx]

        if not current_class:
            update_detection(name, roll, class_info, "recognized", f"👤 Recognized: {name} (No Class)")
            return jsonify({"message": f"👤 Recognized: {name} (No Class)", "color": "cyan", "current_class": class_info})

        # Mark Attendance
        cursor.execute("SELECT id, email FROM students WHERE roll_no=%s", (roll,))
        student = cursor.fetchone()
        if student:
            cursor.execute(
                "SELECT id FROM attendance WHERE student_id=%s AND date=%s AND class_id=%s",
                (student['id'], date_today, current_class['id'])
            )
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO attendance (student_id, date, time, status, class_id, method) VALUES (%s, %s, %s, 'Present', %s, 'auto')",
                    (student['id'], date_today, time_now, current_class['id'])
                )
                db.commit()

                if student.get('email'):
                    send_attendance_emails_in_background([{
                        'student_email': student['email'],
                        'student_name': name,
                        'status': 'Present',
                        'subject': current_class['subject_name'],
                        'date': date_today,
                        'time': time_now
                    }])

                message = f"✅ Present: {name}"
                update_detection(name, roll, current_class['subject_name'], "present", message)
                return jsonify({"message": message, "color": "green", "current_class": class_info})
            else:
                message = f"ℹ️ Already Marked: {name}"
                update_detection(name, roll, current_class['subject_name'], "already_attended", message)
                return jsonify({"message": message, "color": "blue", "current_class": class_info})

        return jsonify({"message": "Student DB Error", "color": "red", "current_class": class_info})

    except Exception as e:
        print(f"❌ Error in process_frame: {e}")
        return jsonify({"message": f"Server Error: {e}", "color": "red", "current_class": "Error"})
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()


# ==================================================
# 🧪 EMAIL TEST ROUTES
# ==================================================
@app.route('/test_college_email')
def test_college_email():
    """Test the college email system"""
    try:
        msg = Message(
            subject="🎓 Khushal Degree College - Email System Active!",
            recipients=[os.environ.get('MAIL_USERNAME', 'khushaldegreecollege@gmail.com')],
            body="""
🎉 CONGRATULATIONS!

Your Khushal Degree College Face Attendance System
email notification system is now fully operational!

Best regards,
Face Attendance System
Khushal Degree College
            """
        )
        mail.send(msg)
        return f"✅ College email system ACTIVATED successfully! Check {os.environ.get('MAIL_USERNAME', 'khushaldegreecollege@gmail.com')}"
    except Exception as e:
        return f"❌ Email test failed: {str(e)}"


# ==================================================
# 🕒 AUTO-ABSENT SCHEDULER (STRICT & SMART)
# ==================================================

def mark_absentees_job():
    """
    Runs every 1 minute.
    Checks for classes that ended within the last 5 minutes.
    Marks anyone NOT Present and NOT on Leave as 'Absent' immediately.
    """
    db = get_db_connection()
    if not db:
        return

    try:
        cursor = db.cursor(dictionary=True)
        # ⚠️ TIMEZONE FIX: Hamesha Pakistan (PKT) time use karega
        now = datetime.utcnow() + timedelta(hours=5)
        date_today = now.date()

        current_time = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")

        time_window_start = (now - timedelta(minutes=5)).strftime("%H:%M:%S")

        cursor.execute("""
            SELECT * FROM classes
            WHERE day_of_week = %s
            AND end_time <= %s
            AND end_time > %s
        """, (day_name, current_time, time_window_start))

        ended_classes = cursor.fetchall()
        if not ended_classes:
            return

        for cls in ended_classes:
            class_id = cls['id']
            semester = cls['semester']
            subject = cls['subject_name']
            class_end = cls['end_time']

            print(f"🏁 Class Ended: {subject} ({semester}) at {class_end}. Marking absentees...")

            cursor.execute("""
                SELECT id, name, email FROM students
                WHERE semester = %s
                AND status = 'approved'
                AND id NOT IN (
                    SELECT student_id FROM attendance
                    WHERE date = %s AND class_id = %s
                )
                AND id NOT IN (
                    SELECT student_id FROM leaves
                    WHERE status = 'Approved'
                    AND %s BETWEEN start_date AND end_date
                )
            """, (semester, date_today, class_id, date_today))

            absentees = cursor.fetchall()

            email_list = []
            for student in absentees:
                cursor.execute("""
                    INSERT INTO attendance (student_id, date, time, status, class_id, method)
                    VALUES (%s, %s, %s, 'Absent', %s, 'auto')
                """, (student['id'], date_today, class_end, class_id))

                print(f"❌ Marked Absent: {student['name']}")

                if student['email']:
                    email_list.append({
                        'student_email': student['email'],
                        'student_name': student['name'],
                        'status': 'Absent',
                        'subject': subject,
                        'date': date_today,
                        'time': class_end
                    })

            db.commit()

            if email_list:
                send_attendance_emails_in_background(email_list)

    except Exception as e:
        print(f"💥 Scheduler Error: {e}")
    finally:
        if db:
            db.close()


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(mark_absentees_job, 'interval', minutes=1)
scheduler.start()


# ==================================================
# 🎓 STUDENT SIGNUP (3-Angle, YOLO + PyTorch Embeddings)
# ==================================================
@app.route('/student_signup', methods=['GET', 'POST'])
def student_signup():
    db = None
    if request.method == 'POST':
        try:
            name = request.form['name']
            roll_no = request.form['roll_no']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            semester = request.form.get('semester', '1st Semester')

            b64_front = request.form.get('img_front')
            b64_left = request.form.get('img_left')
            b64_right = request.form.get('img_right')

            if password != confirm_password:
                return render_template('student_signup.html', error="Passwords do not match!")

            if not (b64_front and b64_left and b64_right):
                return render_template('student_signup.html', error="Please capture all 3 angles.")

            angles = {"FRONT": b64_front, "LEFT": b64_left, "RIGHT": b64_right}
            embeddings = {}

            for label, b64_img in angles.items():
                img = decode_b64_image(b64_img)
                if img is None:
                    return render_template('student_signup.html', error=f"Could not decode {label} image. Please try again.")

                # 1. Blur check
                is_blur, score = is_image_blurry(img, threshold=50)
                if is_blur:
                    return render_template('student_signup.html', error=f"{label} photo is too blurry (Score: {int(score)}). Hold steady!")

                # 2. YOLO face detection + size check
                crop, err = detect_and_crop_face(img)
                if err:
                    return render_template('student_signup.html', error=f"{label}: {err}")

                # 3. Generate embedding
                embeddings[label] = get_face_embedding(crop)

            # Identity sanity check between angles (lenient - just catches totally different faces)
            sim_left = cosine_similarity(embeddings["FRONT"], embeddings["LEFT"])
            sim_right = cosine_similarity(embeddings["FRONT"], embeddings["RIGHT"])
            if sim_left < 0.3 or sim_right < 0.3:
                return render_template('student_signup.html', error="⚠️ The 3 photos don't seem to match the same person. Please retake all 3 angles.")

            student_encodings = [
                embeddings["FRONT"].tolist(),
                embeddings["LEFT"].tolist(),
                embeddings["RIGHT"].tolist(),
            ]
            face_data_json = json.dumps(student_encodings)

            db = get_db_connection()
            if not db:
                return render_template('student_signup.html', error="Database connection error")
            cursor = db.cursor(dictionary=True)

            cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
            if cursor.fetchone():
                return render_template('student_signup.html', error="Roll Number already exists!")

            hashed_pw = generate_password_hash(password)

            cursor.execute("""
                INSERT INTO students (name, roll_no, email, password, semester, status, face_data)
                VALUES (%s, %s, %s, %s, %s, 'pending', %s)
            """, (name, roll_no, email, hashed_pw, semester, face_data_json))

            db.commit()
            load_known_faces()

            return render_template('student_signup.html', message="✅ Registration Successful! Pending Admin Approval.")

        except Exception as e:
            return render_template('student_signup.html', error=f"Error: {e}")
        finally:
            if db:
                db.close()

    return render_template('student_signup.html')


# ==================================================
# 🏠 MAIN ROUTES
# ==================================================

@app.route('/')
def index():
    if session.get('role') == 'admin':
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('role') == 'admin':
        return redirect(url_for('index'))
    if request.method == 'POST':
        user, pwd = request.form['username'], request.form['password']
        if user == os.environ.get('ADMIN_USER', 'admin') and pwd == os.environ.get('ADMIN_PASS', 'admin123'):
            session['logged_in'], session['role'] = True, 'admin'
            return redirect(url_for('index'))
        return render_template('login.html', error="❌ Invalid credentials")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ==================================================
# 📊 DASHBOARD & STATS
# ==================================================

@app.route('/dashboard_stats')
def dashboard_stats():
    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = db.cursor(dictionary=True)

        # 1. Total Approved Students
        cursor.execute("SELECT COUNT(*) AS total FROM students WHERE status='approved'")
        student_count = cursor.fetchone()
        total_students = student_count['total'] if student_count else 0

       # 2. Present Today
        now_pkt = datetime.utcnow() + timedelta(hours=5)
        today = now_pkt.date()
        cursor.execute("SELECT COUNT(DISTINCT student_id) AS present_today FROM attendance WHERE date = %s AND status = 'Present'", (today,))
        present_result = cursor.fetchone()
        present_today = present_result['present_today'] if present_result else 0

        # 3. Upcoming Class (Smart Weekly Wrapping Logic)
        now = now_pkt
        current_time = now.time()
        current_day = now.strftime("%A")

        days_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }

        cursor.execute("SELECT subject_name, day_of_week, start_time FROM classes")
        all_classes = cursor.fetchall()

        upcoming_class_name = "No Classes"
        if all_classes and current_day in days_map:
            current_day_idx = days_map[current_day]
            min_diff = None
            upcoming_cls = None

            for cls in all_classes:
                day = cls['day_of_week']
                if day not in days_map:
                    continue
                day_idx = days_map[day]

                start_t = cls['start_time']
                if isinstance(start_t, timedelta):
                    start_t = (datetime.min + start_t).time()
                elif isinstance(start_t, str):
                    try:
                        start_t = datetime.strptime(start_t, "%H:%M:%S").time()
                    except ValueError:
                        try:
                            start_t = datetime.strptime(start_t, "%H:%M").time()
                        except ValueError:
                            continue

                day_diff = (day_idx - current_day_idx) % 7
                if day_diff == 0:
                    if start_t <= current_time:
                        day_diff = 7

                diff_seconds = day_diff * 86400 + (start_t.hour - current_time.hour) * 3600 + (start_t.minute - current_time.minute) * 60 + (start_t.second - current_time.second)

                if min_diff is None or diff_seconds < min_diff:
                    min_diff = diff_seconds
                    upcoming_cls = cls
                    upcoming_cls['resolved_start_time'] = start_t

            if upcoming_cls:
                t = upcoming_cls['resolved_start_time']
                time_str = t.strftime("%I:%M %p").lstrip('0')
                day_str = upcoming_cls['day_of_week']

                day_diff = (days_map[day_str] - current_day_idx) % 7
                if day_diff == 0:
                    day_label = "Today" if t > current_time else f"Next {day_str}"
                elif day_diff == 1:
                    day_label = "Tomorrow"
                else:
                    day_label = day_str

                upcoming_class_name = f"{upcoming_cls['subject_name']} ({day_label} {time_str})"

        # 4. Pending Signups (Students + Professors)
        cursor.execute("SELECT COUNT(*) as count FROM students WHERE status='pending'")
        s_pending = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM professors WHERE status='pending'")
        p_pending = cursor.fetchone()['count']

        total_pending_signups = s_pending + p_pending

        return jsonify({
            "students": total_students,
            "present_today": present_today,
            "upcoming_class": upcoming_class_name,
            "pending_signups": total_pending_signups
        })
    except Exception as e:
        print("❌ Dashboard stats error:", e)
        return jsonify({"students": 0, "present_today": 0, "upcoming_class": "Error", "pending_signups": 0})
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()


# ==================================================
# 🎓 STUDENT MANAGEMENT
# ==================================================

@app.route('/manage_students')
@admin_required
def manage_students():
    sem_filter = request.args.get('semester')
    db = None
    cursor = None
    students = []
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500

        cursor = db.cursor(dictionary=True)

        sql = "SELECT id, name, roll_no, email, semester FROM students WHERE status='approved'"
        params = []
        if sem_filter and sem_filter != "All":
            sql += " AND semester = %s"
            params.append(sem_filter)

        sql += " ORDER BY roll_no"
        cursor.execute(sql, tuple(params))
        students = cursor.fetchall()
    except Exception as e:
        print(f"❌ Error managing students: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()
    return render_template('manage_students.html', students=students, selected_semester=sem_filter or "All")


@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@admin_required
def edit_student(student_id):
    db = None
    cursor = None
    student = None
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)

        if request.method == 'POST':
            name = request.form['name']
            roll_no = request.form['roll_no']
            email = request.form['email']
            semester = request.form.get('semester', '1st Semester')
            cursor.execute(
                "UPDATE students SET name=%s, roll_no=%s, email=%s, semester=%s WHERE id=%s",
                (name, roll_no, email, semester, student_id)
            )
            db.commit()
            load_known_faces()
            return redirect(url_for('manage_students'))

        cursor.execute("SELECT id, name, roll_no, email, semester, status FROM students WHERE id = %s", (student_id,))
        student = cursor.fetchone()

    except Exception as e:
        if db:
            db.rollback()
        print(f"❌ Error editing student: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    if not student:
        return "Student not found", 404

    return render_template('edit_student.html', student=student)


@app.route('/delete_student/<int:student_id>')
@admin_required
def delete_student(student_id):
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor()
    cursor.execute("DELETE FROM students WHERE id=%s", (student_id,))
    db.commit()
    cursor.close()
    db.close()
    load_known_faces()
    return redirect(url_for('manage_students'))


# ==================================================
# 🎓 STUDENT AUTHENTICATION
# ==================================================

@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if session.get('role') == 'student':
        return redirect(url_for('student_dashboard'))
    if request.method == 'POST':
        db = get_db_connection()
        if not db:
            return render_template('student_login.html', error="DB error")
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE roll_no=%s", (request.form['roll_no'],))
        st = cursor.fetchone()
        cursor.close()
        db.close()

        if st and st['password'] and check_password_hash(st['password'], request.form['password']):
            if st['status'] == 'approved':
                session.update({'logged_in': True, 'role': 'student', 'user_id': st['id'], 'name': st['name']})
                return redirect(url_for('student_dashboard'))
            elif st['status'] == 'pending':
                return render_template('student_login.html', error="⏳ Account pending approval.")
            return render_template('student_login.html', error="❌ Account rejected.")
        return render_template('student_login.html', error="Invalid credentials.")
    return render_template('student_login.html')


@app.route('/student_dashboard')
def student_dashboard():
    if session.get('role') != 'student':
        return redirect(url_for('student_login'))
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT c.subject_name, COUNT(a.id) as total_classes,
               SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) as presents,
               ROUND(SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) * 100.0 / COUNT(a.id), 1) as percentage
        FROM attendance a JOIN classes c ON a.class_id=c.id
        WHERE a.student_id=%s GROUP BY c.subject_name
    """, (session['user_id'],))
    attendance_data = cursor.fetchall()

    cursor.execute("SELECT * FROM leaves WHERE student_id=%s ORDER BY created_at DESC", (session['user_id'],))
    leaves = cursor.fetchall()
    cursor.close()
    db.close()

    grouped_leaves = {}
    for r in leaves:
        d = r['created_at'].strftime('%A, %B %d, %Y') if r['created_at'] else "Unknown"
        grouped_leaves.setdefault(d, []).append(r)

    return render_template('student_dashboard.html', attendance_data=attendance_data, grouped_leaves=grouped_leaves, student_name=session['name'])


@app.route('/student_logout')
def student_logout():
    session.clear()
    return redirect(url_for('student_login'))


# ==================================================
# 📝 LEAVE MANAGEMENT
# ==================================================

@app.route('/apply_leave', methods=['GET', 'POST'])
def apply_leave():
    if session.get('role') != 'student':
        return redirect(url_for('login'))

    logged_in_student_id = session['user_id']

    db = None
    cursor = None
    student = None
    classes = []

    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT id, name, roll_no, email FROM students WHERE id = %s", (logged_in_student_id,))
        student = cursor.fetchone()
        cursor.execute("SELECT subject_name FROM classes")
        classes = cursor.fetchall()

        if request.method == 'POST':
            subject_name = request.form.get('subject_name', None)
            application_purpose = request.form['application_purpose']
            application_text = request.form['application_text']
            start_date = request.form['start_date']
            end_date = request.form['end_date']

            if subject_name:
                cursor.execute(
                    "INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')",
                    (logged_in_student_id, subject_name, application_purpose, application_text, start_date, end_date)
                )
            else:
                cursor.execute("SELECT semester FROM students WHERE id = %s", (logged_in_student_id,))
                student_data_row = cursor.fetchone()

                if student_data_row:
                    student_semester = student_data_row['semester']
                    cursor.execute("SELECT DISTINCT subject_name FROM classes WHERE semester = %s", (student_semester,))
                    semester_subjects = cursor.fetchall()

                    if semester_subjects:
                        for sub in semester_subjects:
                            cursor.execute(
                                "INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')",
                                (logged_in_student_id, sub['subject_name'], application_purpose, application_text, start_date, end_date)
                            )
                    else:
                        cursor.execute(
                            "INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')",
                            (logged_in_student_id, None, application_purpose, application_text, start_date, end_date)
                        )
                else:
                    cursor.execute(
                        "INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')",
                        (logged_in_student_id, None, application_purpose, application_text, start_date, end_date)
                    )

            db.commit()

            if student and student['email']:
                def send_leave_email_async(app_context, student_email, student_name, subject_name, purpose, start, end):
                    with app_context:
                        try:
                            msg = Message(
                                subject="📝 Leave Application Submitted Successfully",
                                recipients=[student_email],
                                body=f"""
Dear {student_name},

Your leave application has been submitted successfully and is pending approval.

📚 Subject: {subject_name or 'All Subjects'}
🎯 Purpose: {purpose}
📅 Period: {start} to {end}

Best regards,
Face Attendance System
Khushal Degree College
                                """
                            )
                            mail.send(msg)
                        except Exception as e:
                            print(f"❌ Failed to send leave confirmation email: {e}")

                thread = threading.Thread(
                    target=send_leave_email_async,
                    args=(app.app_context(), student['email'], student['name'], subject_name, application_purpose, start_date, end_date)
                )
                thread.daemon = True
                thread.start()

            return render_template('apply_leave.html', student=student, classes=classes, message="✅ Leave application submitted successfully!")

    except Exception as e:
        if db and request.method == 'POST':
            db.rollback()
        print(f"❌ Error applying for leave: {e}")
        return render_template('apply_leave.html', student=student, classes=classes, message=f"❌ Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    return render_template('apply_leave.html', student=student, classes=classes)


# ==========================================
# 🔔 ADMIN: View Signups (Students & Professors)
# ==========================================
@app.route('/view_requests', methods=['GET', 'POST'])
@admin_required
def view_requests():
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        req_type = request.form.get('type')
        action = request.form.get('action')

        if req_type == 'student':
            sid = request.form.get('student_id')
            status = 'approved' if action == 'approve' else 'rejected'
            if action == 'reject':
                cursor.execute("DELETE FROM students WHERE id=%s", (sid,))
            else:
                cursor.execute("UPDATE students SET status=%s WHERE id=%s", (status, sid))
                load_known_faces()

        elif req_type == 'professor':
            pid = request.form.get('professor_id')
            status = 'approved' if action == 'approve' else 'rejected'
            if action == 'reject':
                cursor.execute("DELETE FROM professors WHERE id=%s", (pid,))
            else:
                cursor.execute("UPDATE professors SET status=%s WHERE id=%s", (status, pid))

        db.commit()
        cursor.close()
        db.close()
        return redirect(url_for('view_requests'))

    cursor.execute("SELECT * FROM students WHERE status='pending'")
    pending_students = cursor.fetchall()

    cursor.execute("SELECT * FROM professors WHERE status='pending'")
    pending_professors = cursor.fetchall()

    cursor.close()
    db.close()
    return render_template('view_requests.html', pending_students=pending_students, pending_professors=pending_professors)


# ==================================================
# 👨‍🏫 PROFESSOR MANAGEMENT
# ==================================================

@app.route('/professor_signup', methods=['GET', 'POST'])
def professor_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm_password']

        if password != confirm:
            return render_template('professor_signup.html', error="❌ Passwords do not match!")

        hashed_pw = generate_password_hash(password)

        db = get_db_connection()
        if not db:
            return render_template('professor_signup.html', error="Database connection error")
        cursor = db.cursor()
        try:
            cursor.execute("SELECT id FROM professors WHERE email=%s", (email,))
            if cursor.fetchone():
                return render_template('professor_signup.html', error="❌ Email already registered!")

            cursor.execute(
                "INSERT INTO professors (name, email, password, status) VALUES (%s, %s, %s, 'pending')",
                (name, email, hashed_pw)
            )
            db.commit()
            return render_template('professor_signup.html', message="✅ Application Sent! Please wait for Admin approval.")
        except Exception as e:
            return render_template('professor_signup.html', error=f"Error: {e}")
        finally:
            cursor.close()
            db.close()

    return render_template('professor_signup.html')


@app.route('/professor_login', methods=['GET', 'POST'])
def professor_login():
    if session.get('role') == 'professor':
        return redirect(url_for('professor_dashboard'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        db = get_db_connection()
        if not db:
            return render_template('professor_login.html', error="Database connection error")
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM professors WHERE email=%s", (email,))
        prof = cursor.fetchone()
        cursor.close()
        db.close()

        if prof:
            if prof['status'] != 'approved':
                return render_template('professor_login.html', error="⏳ Account pending approval.")
            if prof['password'] and (check_password_hash(prof['password'], password) or prof['password'] == password):
                session['logged_in'] = True
                session['role'] = 'professor'
                session['user_id'] = prof['id']
                session['name'] = prof['name']
                return redirect(url_for('professor_dashboard'))
            elif not prof['password']:
                return redirect(url_for('professor_set_password', professor_id=prof['id'], email=prof['email']))

        return render_template('professor_login.html', error="❌ Invalid credentials")
    return render_template('professor_login.html')


@app.route('/professor_set_password', methods=['GET', 'POST'])
def professor_set_password():
    if request.method == 'GET':
        professor_id = request.args.get('professor_id')
        email = request.args.get('email')
        if not professor_id or not email:
            return redirect(url_for('professor_login'))
        return render_template('professor_set_password.html', professor_id=professor_id, email=email)

    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return render_template('professor_set_password.html', error="Database connection error", **request.form)
        cursor = db.cursor(dictionary=True)

        professor_id = request.form['professor_id']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form['email']

        if password != confirm_password:
            return render_template('professor_set_password.html', error="Passwords do not match!", professor_id=professor_id, email=email)

        if len(password) < 6:
            return render_template('professor_set_password.html', error="Password must be at least 6 characters!", professor_id=professor_id, email=email)

        hashed_password = generate_password_hash(password)

        cursor.execute("UPDATE professors SET password = %s WHERE id = %s", (hashed_password, professor_id))
        db.commit()

        cursor.execute("SELECT * FROM professors WHERE id = %s", (professor_id,))
        professor = cursor.fetchone()

        session['logged_in'] = True
        session['role'] = 'professor'
        session['user_id'] = professor['id']
        session['name'] = professor['name']

        return redirect(url_for('professor_dashboard'))

    except Exception as e:
        if db:
            db.rollback()
        print(f"❌ Error setting professor password: {e}")
        return render_template('professor_set_password.html', error=f"An error occurred: {e}", **request.form)
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()


@app.route('/manage_professors')
@admin_required
def manage_professors():
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM professors WHERE status='approved' ORDER BY name")
    professors = cursor.fetchall()
    cursor.close()
    db.close()
    return render_template('manage_professors.html', professors=professors)


@app.route('/edit_professor/<int:id>', methods=['GET', 'POST'])
@admin_required
def edit_professor(id):
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        cursor.execute("UPDATE professors SET name=%s, email=%s WHERE id=%s", (name, email, id))
        db.commit()
        cursor.close()
        db.close()
        return redirect(url_for('manage_professors'))

    cursor.execute("SELECT * FROM professors WHERE id=%s", (id,))
    professor = cursor.fetchone()
    cursor.close()
    db.close()

    if not professor:
        return "Professor not found", 404

    return render_template('edit_professor.html', professor=professor)


@app.route('/delete_professor/<int:id>')
@admin_required
def delete_professor(id):
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor()
    cursor.execute("DELETE FROM professors WHERE id=%s", (id,))
    db.commit()
    cursor.close()
    db.close()
    return redirect(url_for('manage_professors'))


@app.route('/professor_dashboard')
@professor_required
def professor_dashboard():
    professor_id = session['user_id']

    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT name, email FROM professors WHERE id = %s", (professor_id,))
    prof_data = cursor.fetchone()

    today_name = datetime.now().strftime("%A")
    cursor.execute("""
        SELECT * FROM classes
        WHERE professor_id = %s AND day_of_week = %s
        ORDER BY start_time ASC
    """, (professor_id, today_name))
    todays_classes = cursor.fetchall()

    cursor.execute("""
        SELECT COUNT(*) as count
        FROM attendance a
        JOIN classes c ON a.class_id = c.id
        WHERE c.professor_id = %s
        AND a.date = CURDATE()
        AND a.status = 'Present'
    """, (professor_id,))
    present_count = cursor.fetchone()['count']

    cursor.execute("""
        SELECT COUNT(DISTINCT l.id) as count
        FROM leaves l
        JOIN students s ON l.student_id = s.id
        WHERE l.status = 'Pending'
        AND (
            l.subject_name IN (SELECT subject_name FROM classes WHERE professor_id = %s)
            OR (
                (l.subject_name IS NULL OR l.subject_name = '')
                AND s.semester IN (SELECT semester FROM classes WHERE professor_id = %s)
            )
        )
    """, (professor_id, professor_id))
    leaves_count = cursor.fetchone()['count']

    cursor.execute("SELECT * FROM classes WHERE professor_id = %s", (professor_id,))
    prof_all_classes = cursor.fetchall()

    next_class = None
    if prof_all_classes:
        min_diff = None
        current_time = datetime.now().time()
        current_day = datetime.now().strftime("%A")

        days_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }

        if current_day in days_map:
            current_day_idx = days_map[current_day]

            for cls in prof_all_classes:
                day = cls['day_of_week']
                if day not in days_map:
                    continue
                day_idx = days_map[day]

                start_t = cls['start_time']
                if isinstance(start_t, timedelta):
                    start_t = (datetime.min + start_t).time()
                elif isinstance(start_t, str):
                    try:
                        start_t = datetime.strptime(start_t, "%H:%M:%S").time()
                    except ValueError:
                        try:
                            start_t = datetime.strptime(start_t, "%H:%M").time()
                        except ValueError:
                            continue

                day_diff = (day_idx - current_day_idx) % 7
                if day_diff == 0:
                    if start_t <= current_time:
                        day_diff = 7

                diff_seconds = day_diff * 86400 + (start_t.hour - current_time.hour) * 3600 + (start_t.minute - current_time.minute) * 60 + (start_t.second - current_time.second)

                if min_diff is None or diff_seconds < min_diff:
                    min_diff = diff_seconds
                    next_class = cls
                    next_class['resolved_start_time'] = start_t

            if next_class:
                t = next_class['resolved_start_time']
                time_str = t.strftime("%I:%M %p").lstrip('0')
                day_str = next_class['day_of_week']

                day_diff = (days_map[day_str] - current_day_idx) % 7
                if day_diff == 0:
                    day_label = "Today" if t > current_time else f"Next {day_str}"
                elif day_diff == 1:
                    day_label = "Tomorrow"
                else:
                    day_label = day_str

                next_class['display_time'] = f"{day_label} at {time_str}"

    cursor.close()
    db.close()

    return render_template('professor_dashboard.html', professor=prof_data, classes=todays_classes,
                           present_count=present_count, leaves_count=leaves_count, next_class=next_class)


@app.route('/professor_logout')
def professor_logout():
    session.clear()
    return redirect(url_for('professor_login'))


@app.route('/professor_leaves', methods=['GET', 'POST'])
@professor_required
def professor_leaves():
    professor_id = session['user_id']
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        leave_id = request.form.get('leave_id')
        action = request.form.get('action')

        cursor.execute("UPDATE leaves SET status=%s WHERE id=%s", (action, leave_id))
        db.commit()
        cursor.close()
        db.close()
        flash(f"Leave {action} successfully!", "success")
        return redirect(url_for('professor_leaves'))

    cursor.execute("""
        SELECT DISTINCT l.*, s.name, s.roll_no, s.semester
        FROM leaves l
        JOIN students s ON l.student_id = s.id
        WHERE l.status = 'Pending'
        AND (
            l.subject_name IN (
                SELECT subject_name FROM classes WHERE professor_id = %s
            )
            OR (
                (l.subject_name IS NULL OR l.subject_name = '')
                AND s.semester IN (
                    SELECT semester FROM classes WHERE professor_id = %s
                )
            )
        )
        ORDER BY l.start_date DESC
    """, (professor_id, professor_id))

    leave_records = cursor.fetchall()
    cursor.close()
    db.close()

    return render_template('professor_leaves.html', leave_records=leave_records)


@app.route('/professor_approve_leave', methods=['POST'])
@professor_required
def professor_approve_leave():
    leave_id = request.form['leave_id']
    action = request.form['action']  # 'Approved' or 'Rejected'

    db = get_db_connection()
    if not db:
        return jsonify({'success': False, 'error': 'Database connection error'})
    cursor = db.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT l.*, s.name, s.email, s.semester
            FROM leaves l
            JOIN students s ON l.student_id = s.id
            WHERE l.id = %s
        """, (leave_id,))
        leave = cursor.fetchone()

        if not leave:
            return jsonify({'success': False, 'error': 'Leave not found'})

        cursor.execute("UPDATE leaves SET status = %s WHERE id = %s", (action, leave_id))

        if action == 'Approved':
            student_id = leave['student_id']
            subject_name = leave['subject_name']
            semester = leave['semester']
            start_date = leave['start_date']
            end_date = leave['end_date']

            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

            current_date = start_date
            while current_date <= end_date:
                if subject_name:
                    cursor.execute("SELECT id FROM classes WHERE subject_name = %s AND semester = %s", (subject_name, semester))
                else:
                    cursor.execute("SELECT id FROM classes WHERE semester = %s", (semester,))

                target_classes = cursor.fetchall()

                for cls in target_classes:
                    class_id = cls['id']

                    cursor.execute("DELETE FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, class_id, current_date))

                    cursor.execute("""
                        INSERT INTO attendance (student_id, class_id, date, time, status, method)
                        VALUES (%s, %s, %s, NOW(), 'Leave', 'system')
                    """, (student_id, class_id, current_date))

                current_date += timedelta(days=1)

        db.commit()

        if leave['email']:
            try:
                send_leave_status_notification(
                    leave['email'], leave['name'], action, leave['subject_name'] or 'All Subjects',
                    leave['start_date'], leave['end_date'], leave.get('application_purpose')
                )
            except Exception:
                pass

        return jsonify({'success': True, 'message': f'Leave {action} and attendance updated!'})

    except Exception as e:
        if db:
            db.rollback()
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cursor.close()
        db.close()


# ==================================================
# 🏫 CLASS MANAGEMENT
# ==================================================

@app.route('/manage_classes', methods=['GET', 'POST'])
@admin_required
def manage_classes():
    db = None
    cursor = None
    professors = []
    classes = []
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT id, name FROM professors WHERE status='approved'")
        professors = cursor.fetchall()

        if request.method == 'POST':
            subject_name = request.form['subject_name']
            professor_id = request.form['professor_id']
            semester = request.form.get('semester', '1st Semester')
            day_of_week = request.form['day_of_week']
            start_time = request.form['start_time']
            end_time = request.form['end_time']
            cursor.execute(
                "INSERT INTO classes (subject_name, professor_id, semester, day_of_week, start_time, end_time) VALUES (%s, %s, %s, %s, %s, %s)",
                (subject_name, professor_id, semester, day_of_week, start_time, end_time)
            )
            db.commit()
            return redirect(url_for('manage_classes'))

        cursor.execute("""
            SELECT c.id, c.subject_name, c.semester, p.name AS professor_name, c.day_of_week, c.start_time, c.end_time
            FROM classes c LEFT JOIN professors p ON c.professor_id = p.id
            ORDER BY FIELD(c.day_of_week, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
        """)
        classes = cursor.fetchall()

    except Exception as e:
        if db and request.method == 'POST':
            db.rollback()
        print(f"❌ Error managing classes: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    return render_template('manage_classes.html', professors=professors, classes=classes)


@app.route('/edit_class/<int:class_id>', methods=['GET', 'POST'])
@admin_required
def edit_class(class_id):
    db = get_db_connection()
    if not db:
        return "Database connection error", 500
    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        try:
            subject = request.form['subject_name']
            prof_id = request.form['professor_id']
            semester = request.form['semester']
            day = request.form['day_of_week']
            start = request.form['start_time']
            end = request.form['end_time']

            cursor.execute("""
                UPDATE classes
                SET subject_name=%s, professor_id=%s, semester=%s, day_of_week=%s, start_time=%s, end_time=%s
                WHERE id=%s
            """, (subject, prof_id, semester, day, start, end, class_id))
            db.commit()
            cursor.close()
            db.close()
            return redirect(url_for('manage_classes'))

        except Exception as e:
            print(f"Error updating class: {e}")
            if db:
                db.rollback()

    cursor.execute("SELECT * FROM classes WHERE id=%s", (class_id,))
    class_info = cursor.fetchone()

    cursor.execute("SELECT id, name FROM professors WHERE status='approved'")
    professors = cursor.fetchall()

    cursor.close()
    db.close()

    if not class_info:
        return "Class not found", 404

    def format_time(t):
        if hasattr(t, 'seconds'):
            seconds = t.seconds
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"{h:02}:{m:02}"
        return str(t)

    class_info['start_time'] = format_time(class_info['start_time'])
    class_info['end_time'] = format_time(class_info['end_time'])

    return render_template('edit_class.html', class_info=class_info, professors=professors)


@app.route('/delete_class/<int:class_id>')
@admin_required
def delete_class(class_id):
    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)
        cursor.execute("DELETE FROM classes WHERE id=%s", (class_id,))
        db.commit()
    except Exception as e:
        if db:
            db.rollback()
        print(f"❌ Error deleting class: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    return redirect(url_for('manage_classes'))


# ==================================================
# 📊 ATTENDANCE SYSTEM
# ==================================================

@app.route('/view_attendance')
@admin_required
def view_attendance():
    db = None
    cursor = None
    classes = []
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT DISTINCT subject_name FROM classes")
        classes = cursor.fetchall()
    except Exception as e:
        print(f"❌ Error viewing attendance: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    return render_template('view_attendance.html', classes=classes)


@app.route('/attendance_summary_v2')
def attendance_summary_v2():
    subject = request.args.get('subject', 'all')
    period = request.args.get('period', 'day')
    semester = request.args.get('semester', 'all')

    db = get_db_connection()
    if not db:
        return jsonify([])
    cursor = db.cursor(dictionary=True)

    base_query = """
        SELECT s.name, s.roll_no, s.semester, c.subject_name,
        COUNT(CASE WHEN a.status='Present' THEN 1 END) as presents,
        COUNT(CASE WHEN a.status='Absent' THEN 1 END) as absents,
        COUNT(CASE WHEN a.status='Leave' THEN 1 END) as leaves,
        COUNT(a.id) as total_classes
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id
        LEFT JOIN classes c ON a.class_id = c.id
        WHERE 1=1
    """
    params = []

    if semester != 'all':
        base_query += " AND s.semester = %s"
        params.append(semester)

    if subject != 'all':
        base_query += " AND c.subject_name = %s"
        params.append(subject)

    if period == 'day':
        base_query += " AND a.date = CURDATE()"
    elif period == 'week':
        base_query += " AND YEARWEEK(a.date, 1) = YEARWEEK(CURDATE(), 1)"
    elif period == 'month':
        base_query += " AND MONTH(a.date) = MONTH(CURDATE()) AND YEAR(a.date) = YEAR(CURDATE())"

    base_query += " GROUP BY s.id, c.subject_name ORDER BY s.semester, s.roll_no"

    cursor.execute(base_query, tuple(params))
    data = cursor.fetchall()

    for row in data:
        if row['total_classes'] > 0:
            row['percentage'] = round((row['presents'] / row['total_classes']) * 100, 1)
        else:
            row['percentage'] = 0

    cursor.close()
    db.close()
    return jsonify(data)


@app.route('/get_weekly_attendance')
@admin_required
def get_weekly_attendance():
    semester = request.args.get('semester')
    subject = request.args.get('subject')
    start_date_str = request.args.get('start_date')

    db = get_db_connection()
    if not db:
        return jsonify([])
    cursor = db.cursor(dictionary=True)

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = start_date + timedelta(days=6)

    cursor.execute("SELECT id, name, roll_no FROM students WHERE semester = %s ORDER BY roll_no", (semester,))
    students = cursor.fetchall()

    query = """
        SELECT student_id, date, status, time, c.subject_name
        FROM attendance a
        JOIN classes c ON a.class_id = c.id
        WHERE date BETWEEN %s AND %s
    """
    params = [start_date, end_date]

    if subject != 'all':
        query += " AND c.subject_name = %s"
        params.append(subject)

    cursor.execute(query, tuple(params))
    logs = cursor.fetchall()

    stats_query = """
        SELECT student_id,
               COUNT(CASE WHEN status='Present' THEN 1 END) as p,
               COUNT(*) as t
        FROM attendance a
        JOIN classes c ON a.class_id = c.id
        WHERE c.semester = %s
    """
    stats_params = [semester]

    if subject != 'all':
        stats_query += " AND c.subject_name = %s"
        stats_params.append(subject)

    stats_query += " GROUP BY student_id"
    cursor.execute(stats_query, tuple(stats_params))
    stats_data = {row['student_id']: row for row in cursor.fetchall()}

    attendance_map = {}
    for log in logs:
        sid = log['student_id']
        date_key = str(log['date'])
        if sid not in attendance_map:
            attendance_map[sid] = {}
        attendance_map[sid][date_key] = {'status': log['status'], 'time': str(log['time'])}

    final_data = []
    for s in students:
        stat = stats_data.get(s['id'], {'p': 0, 't': 0})
        pct = round((stat['p'] / stat['t']) * 100) if stat['t'] > 0 else 0

        final_data.append({
            'name': s['name'],
            'roll': s['roll_no'],
            'week_data': attendance_map.get(s['id'], {}),
            'overall_percent': pct
        })

    cursor.close()
    db.close()
    return jsonify(final_data)


@app.route('/get_professor_weekly_attendance')
@professor_required
def get_professor_weekly_attendance():
    semester = request.args.get('semester')
    subject = request.args.get('subject')
    start_date_str = request.args.get('start_date')
    professor_id = session['user_id']

    db = get_db_connection()
    if not db:
        return jsonify([])
    cursor = db.cursor(dictionary=True)

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = start_date + timedelta(days=6)

    cursor.execute("SELECT id, name, roll_no FROM students WHERE semester = %s ORDER BY roll_no", (semester,))
    students = cursor.fetchall()

    query = """
        SELECT a.student_id, a.date, a.status, a.time, c.subject_name
        FROM attendance a
        JOIN classes c ON a.class_id = c.id
        WHERE c.professor_id = %s
        AND a.date BETWEEN %s AND %s
    """
    params = [professor_id, start_date, end_date]

    if subject != 'all':
        query += " AND c.subject_name = %s"
        params.append(subject)

    cursor.execute(query, tuple(params))
    logs = cursor.fetchall()

    stats_query = """
        SELECT student_id,
               COUNT(CASE WHEN status='Present' THEN 1 END) as p,
               COUNT(*) as t
        FROM attendance a
        JOIN classes c ON a.class_id = c.id
        WHERE c.professor_id = %s AND c.semester = %s
    """
    stats_params = [professor_id, semester]

    if subject != 'all':
        stats_query += " AND c.subject_name = %s"
        stats_params.append(subject)

    stats_query += " GROUP BY student_id"
    cursor.execute(stats_query, tuple(stats_params))
    stats_data = {row['student_id']: row for row in cursor.fetchall()}

    attendance_map = {}
    for log in logs:
        sid = log['student_id']
        date_key = str(log['date'])
        if sid not in attendance_map:
            attendance_map[sid] = {}
        attendance_map[sid][date_key] = {'status': log['status'], 'time': str(log['time'])}

    final_data = []
    for s in students:
        stat = stats_data.get(s['id'], {'p': 0, 't': 0})
        pct = round((stat['p'] / stat['t']) * 100) if stat['t'] > 0 else 0

        final_data.append({
            'name': s['name'],
            'roll': s['roll_no'],
            'week_data': attendance_map.get(s['id'], {}),
            'overall_percent': pct
        })

    cursor.close()
    db.close()
    return jsonify(final_data)


@app.route('/professor_attendance')
@professor_required
def professor_attendance():
    professor_id = session.get('user_id')
    db = None
    cursor = None
    professor_subjects = []
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT DISTINCT subject_name FROM classes WHERE professor_id = %s", (professor_id,))
        professor_subjects = cursor.fetchall()
    except Exception as e:
        print(f"❌ Error loading professor attendance page: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    return render_template('professor_attendance.html', subjects=professor_subjects)


@app.route('/professor_attendance_summary')
@professor_required
def professor_attendance_summary():
    subject = request.args.get('subject', 'all')
    semester = request.args.get('semester', 'all')
    professor_id = session['user_id']

    db = get_db_connection()
    if not db:
        return jsonify([])
    cursor = db.cursor(dictionary=True)

    query = """
        SELECT s.name, s.roll_no, s.semester, c.subject_name,
        COUNT(CASE WHEN a.status='Present' THEN 1 END) as presents,
        COUNT(CASE WHEN a.status='Absent' THEN 1 END) as absents,
        COUNT(CASE WHEN a.status='Leave' THEN 1 END) as leaves,
        COUNT(a.id) as total_classes
        FROM students s
        JOIN attendance a ON s.id = a.student_id
        JOIN classes c ON a.class_id = c.id
        WHERE c.professor_id = %s
    """
    params = [professor_id]

    if subject != 'all':
        query += " AND c.subject_name = %s"
        params.append(subject)

    if semester != 'all':
        query += " AND s.semester = %s"
        params.append(semester)

    query += " GROUP BY s.id, c.subject_name ORDER BY s.semester, s.roll_no"

    cursor.execute(query, tuple(params))
    data = cursor.fetchall()

    for row in data:
        if row['total_classes'] > 0:
            row['percentage'] = round((row['presents'] / row['total_classes']) * 100, 1)
        else:
            row['percentage'] = 0

    cursor.close()
    db.close()
    return jsonify(data)


# ==================================================
# 🎥 LIVE ATTENDANCE
# ==================================================

@app.route('/live_attendance')
def live_attendance():
    if session.get('role') not in ['admin', 'professor']:
        return redirect(url_for('login'))
    return render_template('live_attendance.html')


# ==================================================
# 📋 MANUAL ATTENDANCE
# ==================================================

@app.route('/manual_attendance')
def manual_attendance():
    role = session.get('role')
    if role not in ['admin', 'professor']:
        return redirect(url_for('login'))

    db = None
    cursor = None
    classes = []
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)

        if role == 'professor':
            cursor.execute("SELECT * FROM classes WHERE professor_id=%s ORDER BY day_of_week", (session['user_id'],))
        else:
            cursor.execute("SELECT * FROM classes ORDER BY day_of_week")

        classes = cursor.fetchall()
    except Exception as e:
        print(f"❌ Error loading manual attendance page: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('manual_attendance.html', classes=classes, today=today)


@app.route('/professor_manual_attendance')
@professor_required
def professor_manual_attendance():
    professor_id = session.get('user_id')
    db = None
    cursor = None
    professor_classes = []

    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM classes WHERE professor_id = %s ORDER BY subject_name", (professor_id,))
        professor_classes = cursor.fetchall()
    except Exception as e:
        print(f"❌ Error loading professor manual attendance: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    formatted_classes = []
    for cls in professor_classes:
        formatted_class = dict(cls)
        formatted_class['start_time'] = str(formatted_class['start_time'])
        formatted_class['end_time'] = str(formatted_class['end_time'])
        formatted_classes.append(formatted_class)

    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('professor_manual_attendance.html', classes=formatted_classes, today=today)


@app.route('/get_class_students/<int:class_id>')
def get_class_students(class_id):
    """Fetches students belonging to the class's semester, plus existing attendance for the date."""
    db = None
    cursor = None
    students = []
    attendance = {}

    try:
        db = get_db_connection()
        if not db:
            return jsonify({"error": "Database connection error"}), 500
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT semester FROM classes WHERE id=%s", (class_id,))
        cls = cursor.fetchone()
        if not cls:
            return jsonify({'students': []})

        semester = cls['semester']

        cursor.execute("SELECT id, name, roll_no FROM students WHERE semester=%s AND status='approved' ORDER BY name", (semester,))
        students = cursor.fetchall()

        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        cursor.execute("SELECT student_id, status FROM attendance WHERE class_id=%s AND date=%s", (class_id, date))
        attendance = {row['student_id']: row['status'] for row in cursor.fetchall()}

    except Exception as e:
        print(f"❌ Error getting class students: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

    return jsonify({'students': students, 'existing_attendance': attendance})


@app.route('/save_manual_attendance', methods=['POST'])
def save_manual_attendance():
    data = request.json
    class_id = data['class_id']
    date = data['date']
    attendance_data = data['attendance']

    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return jsonify({'success': False, 'error': 'Database connection error'})
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT start_time, subject_name FROM classes WHERE id = %s", (class_id,))
        class_info = cursor.fetchone()
        class_time = class_info['start_time']
        subject_name = class_info['subject_name']

        for student_id, status in attendance_data.items():
            cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, class_id, date))
            existing = cursor.fetchone()

            if existing:
                cursor.execute("UPDATE attendance SET status = %s, time = %s, method = 'manual' WHERE id = %s", (status, class_time, existing['id']))
            else:
                cursor.execute("INSERT INTO attendance (student_id, class_id, date, time, status, method) VALUES (%s, %s, %s, %s, %s, 'manual')", (student_id, class_id, date, class_time, status))

        db.commit()

        email_data_list = []
        students_without_email = 0

        for student_id, status in attendance_data.items():
            if status.lower() in ["present", "absent"]:
                cursor.execute("SELECT name, email FROM students WHERE id = %s", (student_id,))
                student_data = cursor.fetchone()

                if student_data and student_data['email']:
                    email_data_list.append({
                        'student_email': student_data['email'],
                        'student_name': student_data['name'],
                        'status': status.capitalize(),
                        'subject': subject_name,
                        'date': date,
                        'time': class_time
                    })
                else:
                    students_without_email += 1

        if email_data_list:
            send_attendance_emails_in_background(email_data_list)
            message = f'Attendance saved for {len(attendance_data)} students. Emails are being sent to {len(email_data_list)} students in background.'
        else:
            message = f'Attendance saved for {len(attendance_data)} students. No emails to send.'

        if students_without_email > 0:
            message += f' ({students_without_email} students without email)'

        return jsonify({'success': True, 'message': message})

    except Exception as e:
        if db:
            db.rollback()
        print(f"❌ Error saving manual attendance: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()


@app.route('/bulk_attendance_action', methods=['POST'])
def bulk_attendance_action():
    data = request.json
    action = data['action']
    student_ids = data['student_ids']
    class_id = data['class_id']
    date = data['date']

    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return jsonify({'success': False, 'error': 'Database connection error'})
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT subject_name FROM classes WHERE id = %s", (class_id,))
        subject_name = cursor.fetchone()['subject_name']
        current_time = datetime.now().strftime('%H:%M:%S')

        for student_id in student_ids:
            cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, class_id, date))
            existing = cursor.fetchone()

            if existing:
                cursor.execute("UPDATE attendance SET status = %s WHERE id = %s", (action, existing['id']))
            else:
                cursor.execute("INSERT INTO attendance (student_id, class_id, date, time, status, method) VALUES (%s, %s, %s, %s, %s, 'manual')", (student_id, class_id, date, current_time, action))

        db.commit()

        if action.lower() in ["present", "absent"]:
            email_data_list = []

            for student_id in student_ids:
                cursor.execute("SELECT name, email FROM students WHERE id = %s", (student_id,))
                student_data = cursor.fetchone()

                if student_data and student_data['email']:
                    email_data_list.append({
                        'student_email': student_data['email'],
                        'student_name': student_data['name'],
                        'status': action.capitalize(),
                        'subject': subject_name,
                        'date': date,
                        'time': current_time
                    })

            if email_data_list:
                send_attendance_emails_in_background(email_data_list)
                return jsonify({'success': True, 'message': f'Bulk {action} applied to {len(student_ids)} students. Emails are being sent in background.'})

        return jsonify({'success': True, 'message': f'Bulk {action} applied to {len(student_ids)} students.'})

    except Exception as e:
        if db:
            db.rollback()
        print(f"❌ Error in bulk attendance: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()


# ==================================================
# 🏁 CLEANUP & RUN
# ==================================================

@atexit.register
def cleanup_on_exit():
    if scheduler.running:
        scheduler.shutdown()
    print("Application exiting. Scheduler stopped.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true')
