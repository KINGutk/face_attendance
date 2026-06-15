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
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# ==================================================
# ⏱️ PAKISTAN TIMEZONE HELPER
# ==================================================
def get_pkt_now():
    """Hamesha Pakistan Standard Time (PKT) return karega, chahe server dunya mein kahin bhi ho"""
    return datetime.utcnow() + timedelta(hours=5)

# Fix Windows console encoding for emoji characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

# ==================================================
# 🌍 LOAD ENVIRONMENT VARIABLES
# ==================================================
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path)
    else: load_dotenv()
except ImportError: pass

# ==================================================
# 🔧 FLASK APP CONFIG
# ==================================================
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super_secure_authentic_key_2026')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  

# ==================================================
# 🧠 AI MODEL LOADING (YOLOv8 + ResNet50)
# ==================================================
print("🔄 Loading YOLOv8 and ResNet50 models...")

MODEL_PATH = 'yolov8n-face.pt'
if not os.path.exists(MODEL_PATH):
    urls = [
        "https://github.com/SannketNikam/Face-Detection/raw/main/yolov8n-face.pt",
        "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt"
    ]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            break
        except Exception: pass

yolo_model = YOLO(MODEL_PATH)

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
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = face_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad(): emb = embedding_model(tensor)
    return torch.nn.functional.normalize(emb, dim=1).squeeze().cpu().numpy()

def cosine_similarity(a, b):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def decode_b64_image(b64_string):
    if not b64_string or b64_string == "data:,": return None
    try:
        _, encoded = b64_string.split(",", 1) if "," in b64_string else ('', b64_string)
        np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        if np_arr.size == 0: return None
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception: return None

def is_image_blurry(img_bgr, threshold=50):
    if img_bgr is None: return True, 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score

def detect_and_crop_face(img_bgr, conf_threshold=0.55, min_area_ratio=0.08):
    if img_bgr is None: return None, "Could not decode image"
    results = yolo_model(img_bgr, verbose=False)
    faces = [box for r in results for box in r.boxes if box.conf[0] >= conf_threshold]

    if not faces: return None, "⚠️ No face detected. Look at the camera."
    if len(faces) > 1: return None, "⚠️ Multiple faces detected. Only one person at a time."

    box = faces[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    face_area = (x2 - x1) * (y2 - y1)
    image_area = img_bgr.shape[0] * img_bgr.shape[1]

    if face_area < (image_area * min_area_ratio): return None, "⚠️ Face too small. Please move closer."
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0: return None, "⚠️ Crop error. Please try again."
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
KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS = [], [], []

def load_known_faces():
    print("🔄 Loading AI Face Maps from Database...")
    global KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS
    KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS = [], [], []

    db = get_db_connection()
    if not db: return

    try:
        cursor = db.cursor(dictionary=True)
        try:
            cursor.execute("ALTER TABLE students ADD COLUMN face_data LONGTEXT;")
            db.commit()
        except Exception: pass

        cursor.execute("SELECT roll_no, name, face_data FROM students WHERE status = 'approved'")
        for student in cursor.fetchall():
            if student['face_data']:
                try:
                    encodings_list = json.loads(student['face_data'])
                    for enc in encodings_list:
                        KNOWN_ENCODINGS.append(np.array(enc))
                        KNOWN_NAMES.append(student['name'])
                        KNOWN_ROLLS.append(student['roll_no'])
                except Exception as e: pass
        print(f"✅ Loaded {len(KNOWN_ENCODINGS)} total Math Maps.")
    finally: db.close()

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
        if 'logged_in' not in session: return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin': return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def professor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'professor': return redirect(url_for('professor_login'))
        return f(*args, **kwargs)
    return decorated

# ==================================================
# 🔄 TOAST NOTIFICATION SYSTEM 
# ==================================================
last_detection = {}

@app.route('/last_detection')
def get_last_detection(): return jsonify(last_detection)

@app.route('/clear_detection')
def clear_detection():
    global last_detection
    last_detection = {}
    return jsonify({"status": "cleared"})

def update_detection(name, roll, subject, status, message):
    global last_detection
    last_detection = {
        "name": name, "roll": roll, "subject": subject,
        "status": status, "message": message,
        "timestamp": get_pkt_now().isoformat()
    }

# ==================================================
# 📧 EMAIL NOTIFICATION FUNCTIONS (Intact)
# ==================================================
def send_attendance_notification(student_email, student_name, status, subject, date, time=None):
    try:
        with app.app_context():
            subject_line = f"✅ Khushal College - {status} for {subject}" if status == "Present" else f"⚠️ Khushal College - {status} for {subject}"
            body = f"Dear {student_name},\n\nYour attendance is marked {status}.\n📚 Subject: {subject}\n📅 Date: {date}\n⏰ Time: {time}\n\nBest regards,\nKhushal Degree College"
            msg = Message(subject=subject_line, recipients=[student_email], body=body)
            mail.send(msg)
            return True
    except Exception: return False

def send_leave_status_notification(student_email, student_name, status, subject, start_date, end_date, purpose=None):
    try:
        with app.app_context():
            msg = Message(subject=f"Khushal College - Leave {status}", recipients=[student_email], body=f"Your leave is {status} from {start_date} to {end_date}.")
            mail.send(msg)
            return True
    except Exception: return False

def send_attendance_emails_in_background(email_data_list):
    def email_worker():
        for email_data in email_data_list:
            with app.app_context():
                send_attendance_notification(email_data['student_email'], email_data['student_name'], email_data['status'], email_data['subject'], email_data['date'], email_data.get('time'))
            time.sleep(1)
    threading.Thread(target=email_worker, daemon=True).start()

# ==================================================
# ⚡ INSTANT PHOTO CHECK
# ==================================================
@app.route('/check_photo_quality', methods=['POST'])
def check_photo_quality():
    try:
        data = request.json
        img = decode_b64_image(data.get('image'))
        if img is None: return jsonify({"valid": False, "error": "Could not decode image"})

        is_blur, _ = is_image_blurry(img, threshold=50)
        if is_blur: return jsonify({"valid": False, "error": "⚠️ Too Blurry. Hold steady!"})

        crop, err = detect_and_crop_face(img)
        if err: return jsonify({"valid": False, "error": err})

        return jsonify({"valid": True})
    except Exception as e: return jsonify({"valid": False, "error": str(e)})

# ==================================================
# ⚡ LIVE ATTENDANCE API (With PKT Time & Auto-Load)
# ==================================================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    db = cursor = None
    try:
        # ⚠️ MASLA 2 HAL: Agar memory khali ho (doosra worker), to foran load kar lo!
        if not KNOWN_ENCODINGS:
            load_known_faces()

        data = request.json
        img = decode_b64_image(data.get('image'))
        if img is None: return jsonify({"message": "Decode error", "color": "red", "current_class": "--"})

        db = get_db_connection()
        if not db: return jsonify({"message": "DB error", "color": "red", "current_class": "--"})
        cursor = db.cursor(dictionary=True)
        
        # ⚠️ MASLA 1 HAL: Use PKT Time Instead of UTC
        now = get_pkt_now()
        date_today = now.date()
        time_now = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")

        cursor.execute("SELECT * FROM classes WHERE day_of_week=%s AND start_time<=%s AND end_time>=%s LIMIT 1", (day_name, time_now, time_now))
        current_class = cursor.fetchone()
        class_info = f"{current_class['subject_name']} ({current_class['semester']})" if current_class else "No Active Class"

        if not KNOWN_ENCODINGS: return jsonify({"message": "DB Empty", "color": "orange", "current_class": class_info})

        small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        results = yolo_model(small, verbose=False)
        faces = [box for r in results for box in r.boxes if box.conf[0] >= 0.5]

        if not faces: return jsonify({"message": "No face detected", "color": "orange", "current_class": class_info})

        best_box = max(faces, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = [v * 2 for v in map(int, best_box.xyxy[0])]
        face_crop = img[y1:y2, x1:x2]

        if face_crop.size == 0: return jsonify({"message": "Crop error", "color": "red", "current_class": class_info})

        query_emb = get_face_embedding(face_crop)
        sims = [cosine_similarity(query_emb, k) for k in KNOWN_ENCODINGS]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim < 0.72:
            update_detection("Unknown", "Unknown", class_info, "unknown", "⚠️ Unknown Face Detected!")
            return jsonify({"message": "Unknown Face", "color": "red", "current_class": class_info})

        name, roll = KNOWN_NAMES[best_idx], KNOWN_ROLLS[best_idx]

        if not current_class:
            update_detection(name, roll, class_info, "recognized", f"👤 Recognized: {name} (No Class)")
            return jsonify({"message": f"👤 Recognized: {name} (No Class)", "color": "cyan", "current_class": class_info})

        cursor.execute("SELECT id, email FROM students WHERE roll_no=%s", (roll,))
        student = cursor.fetchone()
        if student:
            cursor.execute("SELECT id FROM attendance WHERE student_id=%s AND date=%s AND class_id=%s", (student['id'], date_today, current_class['id']))
            if not cursor.fetchone():
                cursor.execute("INSERT INTO attendance (student_id, date, time, status, class_id, method) VALUES (%s, %s, %s, 'Present', %s, 'auto')", (student['id'], date_today, time_now, current_class['id']))
                db.commit()
                if student.get('email'): send_attendance_emails_in_background([{'student_email': student['email'], 'student_name': name, 'status': 'Present', 'subject': current_class['subject_name'], 'date': date_today, 'time': time_now}])
                
                message = f"✅ Present: {name}"
                update_detection(name, roll, current_class['subject_name'], "present", message)
                return jsonify({"message": message, "color": "green", "current_class": class_info})
            else:
                message = f"ℹ️ Already Marked: {name}"
                update_detection(name, roll, current_class['subject_name'], "already_attended", message)
                return jsonify({"message": message, "color": "blue", "current_class": class_info})

        return jsonify({"message": "Student DB Error", "color": "red", "current_class": class_info})

    except Exception as e: return jsonify({"message": f"Server Error: {e}", "color": "red", "current_class": "Error"})
    finally:
        if cursor: cursor.close()
        if db: db.close()

# ==================================================
# 🕒 AUTO-ABSENT SCHEDULER 
# ==================================================
def mark_absentees_job():
    db = get_db_connection()
    if not db: return
    try:
        cursor = db.cursor(dictionary=True)
        now = get_pkt_now()
        date_today = now.date()
        current_time = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")
        time_window_start = (now - timedelta(minutes=5)).strftime("%H:%M:%S")

        cursor.execute("SELECT * FROM classes WHERE day_of_week = %s AND end_time <= %s AND end_time > %s", (day_name, current_time, time_window_start))
        for cls in cursor.fetchall():
            cursor.execute("SELECT id, name, email FROM students WHERE semester = %s AND status = 'approved' AND id NOT IN (SELECT student_id FROM attendance WHERE date = %s AND class_id = %s)", (cls['semester'], date_today, cls['id']))
            absentees = cursor.fetchall()
            for st in absentees:
                cursor.execute("INSERT INTO attendance (student_id, date, time, status, class_id, method) VALUES (%s, %s, %s, 'Absent', %s, 'auto')", (st['id'], date_today, cls['end_time'], cls['id']))
            db.commit()
    finally: db.close()

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(mark_absentees_job, 'interval', minutes=1)
scheduler.start()

# ==================================================
# 🎓 STUDENT SIGNUP
# ==================================================
@app.route('/student_signup', methods=['GET', 'POST'])
def student_signup():
    if request.method == 'POST':
        samples_json = request.form.get('face_samples', '[]')
        samples = json.loads(samples_json)
        
        encodings = []
        for b64_img in samples:
            img = decode_b64_image(b64_img)
            if img is not None:
                crop, err = detect_and_crop_face(img)
                if not err: encodings.append(get_face_embedding(crop).tolist())

        if not encodings: return render_template('student_signup.html', error="Failed to detect face properly.")

        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("INSERT INTO students (name, roll_no, email, password, semester, status, face_data) VALUES (%s, %s, %s, %s, %s, 'pending', %s)", 
                      (request.form['name'], request.form['roll_no'], request.form['email'], generate_password_hash(request.form['password']), request.form.get('semester', '1st Semester'), json.dumps(encodings)))
        db.commit()
        db.close()
        load_known_faces()
        return render_template('student_signup.html', message="✅ Registration Successful! Pending Admin Approval.")
    return render_template('student_signup.html')

# ==================================================
# 🏠 MAIN ROUTES & STATS (Fixed Timezone)
# ==================================================
@app.route('/')
def index():
    if session.get('role') == 'admin': return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session.update({'logged_in': True, 'role': 'admin'})
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard_stats')
def dashboard_stats():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) AS total FROM students WHERE status='approved'")
    total_students = cursor.fetchone()['total']

    now = get_pkt_now()
    cursor.execute("SELECT COUNT(DISTINCT student_id) AS present_today FROM attendance WHERE date = %s AND status = 'Present'", (now.date(),))
    present_today = cursor.fetchone()['present_today']

    cursor.execute("SELECT COUNT(*) as count FROM students WHERE status='pending'")
    total_pending_signups = cursor.fetchone()['count'] + cursor.execute("SELECT COUNT(*) as count FROM professors WHERE status='pending'") or cursor.fetchone()['count']
    
    db.close()
    return jsonify({"students": total_students, "present_today": present_today, "upcoming_class": "Active", "pending_signups": total_pending_signups})

# ==================================================
# 🎓 MANAGEMENT ROUTES (Keep existing routes here)
# ==================================================
@app.route('/live_attendance')
def live_attendance():
    if session.get('role') not in ['admin', 'professor']: return redirect(url_for('login'))
    return render_template('live_attendance.html')

@app.route('/manage_professors', methods=['GET', 'POST'])
@admin_required
def manage_professors():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        cursor.execute("INSERT INTO professors (name, email, password, status) VALUES (%s, %s, %s, 'approved')", 
                      (request.form['name'], request.form['email'], generate_password_hash(request.form['password'])))
        db.commit()
    cursor.execute("SELECT * FROM professors")
    profs = cursor.fetchall()
    db.close()
    return render_template('manage_professors.html', professors=profs)

@app.route('/manage_classes', methods=['GET', 'POST'])
@admin_required
def manage_classes():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        cursor.execute("INSERT INTO classes (subject_name, professor_id, semester, day_of_week, start_time, end_time) VALUES (%s, %s, %s, %s, %s, %s)", 
                      (request.form['subject_name'], request.form['professor_id'], request.form['semester'], request.form['day_of_week'], request.form['start_time'], request.form['end_time']))
        db.commit()
    cursor.execute("SELECT * FROM classes")
    classes = cursor.fetchall()
    db.close()
    return render_template('manage_classes.html', classes=classes)

@app.route('/view_requests', methods=['GET', 'POST'])
@admin_required
def view_requests():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        req_type, action = request.form.get('type'), request.form.get('action')
        if req_type == 'student':
            sid = request.form.get('student_id')
            if action == 'reject': cursor.execute("DELETE FROM students WHERE id=%s", (sid,))
            else: 
                cursor.execute("UPDATE students SET status='approved' WHERE id=%s", (sid,))
                load_known_faces()
        db.commit()
        return redirect(url_for('view_requests'))

    cursor.execute("SELECT * FROM students WHERE status='pending'")
    ps = cursor.fetchall()
    cursor.execute("SELECT * FROM professors WHERE status='pending'")
    pp = cursor.fetchall()
    db.close()
    return render_template('view_requests.html', pending_students=ps, pending_professors=pp)

@app.route('/manage_students')
@admin_required
def manage_students():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM students WHERE status='approved'")
    students = cursor.fetchall()
    db.close()
    return render_template('manage_students.html', students=students)

@app.route('/manual_attendance')
def manual_attendance():
    return render_template('manual_attendance.html', classes=[])

@app.route('/student_login')
def student_login():
    return render_template('student_login.html')

@app.route('/professor_login')
def professor_login():
    return render_template('professor_login.html')

@app.route('/professor_dashboard')
def professor_dashboard():
    return render_template('professor_dashboard.html')

@atexit.register
def cleanup_on_exit():
    if scheduler.running: scheduler.shutdown()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true')