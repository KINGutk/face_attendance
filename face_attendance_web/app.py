from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session, Response, flash
import mysql.connector
import mysql.connector.pooling
from datetime import datetime, timedelta
import os
import sys
import json

# Fix Windows console encoding for emoji characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
        
import base64
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import threading
import time
import atexit
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler

# ==================================================
# 🤖 AI / COMPUTER VISION IMPORTS
# ==================================================
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pickle

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

if os.environ.get('SPACE_ID') and os.path.exists('/data'):
    FACES_DIR = '/data/faces'
else:
    FACES_DIR = os.path.join(app.root_path, "faces")
os.makedirs(FACES_DIR, exist_ok=True)

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super_secure_authentic_key_2026')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ==================================================
# 🧠 AI MODEL LOADING (YOLOv8 + ResNet50)
# ==================================================
print("🔄 Loading YOLOv8 and ResNet50 models...")
import urllib.request
import os

MODEL_PATH = 'yolov8n-face.pt'
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading specialized YOLOv8 Face model from Mirror...")
    
    # 2 Working Mirror links (agar ek fail ho toh doosra chal jaye)
    urls = [
        "https://github.com/SannketNikam/Face-Detection/raw/main/yolov8n-face.pt",
        "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt"
    ]
    
    downloaded = False
    for url in urls:
        try:
            print(f"🔗 Trying to download from: {url}")
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("✅ YOLOv8 Face model downloaded successfully!")
            downloaded = True
            break
        except Exception as e:
            print(f"⚠️ Failed from this link: {e}")
            
    if not downloaded:
        print("❌ All downloads failed. Please check internet or download manually.")

from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pickle

yolo_model = YOLO(MODEL_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = resnet50(pretrained=True)
embedding_model.fc = torch.nn.Linear(2048, 512)  # 512D embeddings
embedding_model = embedding_model.to(device)
embedding_model.eval()

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

known_encodings = []
known_names = []
known_rolls = []
ENCODINGS_CACHE = os.path.join(FACES_DIR, 'encodings_cache.pkl')
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
# 💾 DATABASE CONNECTION POOL
# ==================================================
try:
    db_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name='attendance_pool',
        pool_size=5,
        host=os.environ.get('DB_HOST', 'localhost'),
        port=int(os.environ.get('DB_PORT', 4000)),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASS', ''),
        database=os.environ.get('DB_NAME', 'face_attendance_db'),
        ssl_ca=os.environ.get('DB_SSL_CA', ''),
        ssl_disabled=False
    )
    print("✅ Database Connection Pool initialized")
except Exception as e:
    print(f"❌ DB Pool Error: {e}")

def get_db_connection():
    try:
        return db_pool.get_connection()
    except Exception as e:
        print(f'[DB] Pool error: {e}')
        return None

# ==================================================
# 🔑 AUTHENTICATION WRAPPERS
# ==================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def professor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'professor':
            return redirect(url_for('professor_login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================================================
# 🖼️ ADVANCED FACE RECOGNITION (PyTorch Embeddings)
# ==================================================
def get_face_embedding(img_bgr):
    """Extract 512D embedding from a face crop using ResNet50"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = face_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedding_model(tensor)
    return torch.nn.functional.normalize(emb, dim=1).squeeze().cpu().numpy()

def load_known_faces():
    global known_encodings, known_names, known_rolls
    # Try loading from cache first
    if os.path.exists(ENCODINGS_CACHE):
        with open(ENCODINGS_CACHE, 'rb') as f:
            cache = pickle.load(f)
            known_encodings = cache['encodings']
            known_names     = cache['names']
            known_rolls     = cache['rolls']
        print(f'[FACE] Loaded {len(known_encodings)} encodings from cache')
        return
    _rebuild_encodings()

def _rebuild_encodings():
    global known_encodings, known_names, known_rolls
    enc, names, rolls = [], [], []
    if not os.path.exists(FACES_DIR):
        return
    for folder in os.listdir(FACES_DIR):
        folder_path = os.path.join(FACES_DIR, folder)
        if not os.path.isdir(folder_path): continue
        parts = folder.split('_', 1)
        roll = parts[0]
        name = parts[1].replace('_', ' ') if len(parts) > 1 else folder
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg','.jpeg','.png')): continue
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            results = yolo_model(img, verbose=False)
            for r in results:
                for box in r.boxes:
                    if box.conf[0] < 0.5: continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    face_crop = img[y1:y2, x1:x2]
                    if face_crop.size == 0: continue
                    embedding = get_face_embedding(face_crop)
                    enc.append(embedding)
                    names.append(name)
                    rolls.append(roll)
    known_encodings, known_names, known_rolls = enc, names, rolls
    # Save cache
    with open(ENCODINGS_CACHE, 'wb') as f:
        pickle.dump({'encodings': enc, 'names': names, 'rolls': rolls}, f)
    print(f'[FACE] Built and cached {len(enc)} encodings')

def reload_faces_background():
    """Call this after adding/deleting students — non-blocking"""
    if os.path.exists(ENCODINGS_CACHE):
        os.remove(ENCODINGS_CACHE)
    t = threading.Thread(target=_rebuild_encodings, daemon=True)
    t.start()

# Load faces on initial startup
load_known_faces()

@app.route('/reload_faces')
def reload_faces():
    reload_faces_background()
    return jsonify({"success": True, "message": "Face cache rebuilding in background."})

# ==================================================
# ⚡ INSTANT PHOTO CHECK API (YOLOv8 KYC)
# ==================================================
@app.route('/check_photo_quality', methods=['POST'])
def check_photo_quality():
    data = request.get_json()
    img_data = data.get('image', '')
    if not img_data:
        return jsonify({'valid': False, 'error': 'No image received'})
    try:
        header, encoded = img_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'valid': False, 'error': 'Could not decode image'})

        # 1. Blur check (relaxed for mobile cameras)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 60:
            return jsonify({'valid': False, 'error': f'Image too blurry ({int(blur_score)}). Hold steady.'})

        # 2. YOLO face detection
        results = yolo_model(img, verbose=False)
        faces = []
        for r in results:
            for box in r.boxes:
                if box.conf[0] >= 0.55:
                    faces.append(box)

        # 3. Exactly 1 face required
        if len(faces) == 0:
            return jsonify({'valid': False, 'error': 'No face detected. Position your face in the oval.'})
        if len(faces) > 1:
            return jsonify({'valid': False, 'error': f'{len(faces)} faces detected. Only 1 person allowed.'})

        # 4. Face size check
        box = faces[0]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        face_area = (x2-x1) * (y2-y1)
        frame_area = img.shape[0] * img.shape[1]
        if face_area / frame_area < 0.08:
            return jsonify({'valid': False, 'error': 'Move closer to the camera.'})

        # 5. Confidence check
        if float(box.conf[0]) < 0.65:
            return jsonify({'valid': False, 'error': 'Face unclear. Improve lighting.'})

        return jsonify({'valid': True, 'blur': round(blur_score, 1), 'confidence': round(float(box.conf[0])*100, 1)})

    except Exception as e:
        return jsonify({'valid': False, 'error': f'Error: {str(e)}'})

# ==================================================
# ⚡ LIVE ATTENDANCE API (PyTorch Face Match)
# ==================================================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    img_data = data.get('image', '')
    if not img_data:
        return jsonify({'message': 'No image', 'color': 'red'})
    try:
        header, encoded = img_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'message': 'Could not decode frame', 'color': 'red'})

        # Resize for speed (process at 50% size)
        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        results = yolo_model(small, verbose=False)

        faces = []
        for r in results:
            for box in r.boxes:
                if box.conf[0] >= 0.5:
                    faces.append(box)

        if not faces:
            return jsonify({'message': 'No face detected — center your face', 'color': 'orange'})

        if not known_encodings:
            return jsonify({'message': 'No students enrolled yet', 'color': 'cyan'})

        # Use highest-confidence face
        best_box = max(faces, key=lambda b: float(b.conf[0]))
        x1,y1,x2,y2 = [v*2 for v in map(int, best_box.xyxy[0])]  # scale back to full size
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return jsonify({'message': 'Face crop failed', 'color': 'orange'})

        # Get embedding
        query_emb = get_face_embedding(face_crop)

        # Compare against all known embeddings (cosine similarity)
        import numpy as np
        sims = [float(np.dot(query_emb, k) / (np.linalg.norm(query_emb)*np.linalg.norm(k)+1e-8))
                for k in known_encodings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        confidence = round(best_sim * 100, 1)

        THRESHOLD = 0.72  # cosine similarity threshold
        if best_sim < THRESHOLD:
            return jsonify({'message': f'Unknown face (confidence: {confidence}%)', 'color': 'red', 'confidence': confidence})

        matched_name = known_names[best_idx]
        matched_roll = known_rolls[best_idx]

        # Get current active class and mark attendance
        conn = get_db_connection()
        if not conn:
            return jsonify({'message': 'DB connection failed', 'color': 'red'})
        try:
            cursor = conn.cursor(dictionary=True)
            now = datetime.now()
            current_day  = now.strftime('%A')
            current_time = now.strftime('%H:%M:%S')
            current_date = now.strftime('%Y-%m-%d')

            cursor.execute("""
                SELECT c.id, c.subject_name, s.id as student_id
                FROM classes c
                JOIN students s ON s.roll_no = %s
                WHERE c.day_of_week = %s
                AND %s BETWEEN c.start_time AND DATE_ADD(c.end_time, INTERVAL 15 MINUTE)
            """, (matched_roll, current_day, current_time))
            active_class = cursor.fetchone()

            if not active_class:
                return jsonify({
                    'message': f'✅ {matched_name} recognized (no active class now)',
                    'color': 'cyan',
                    'confidence': confidence,
                    'current_class': 'No active class'
                })

            # Check already marked
            cursor.execute("""
                SELECT id FROM attendance
                WHERE student_id=%s AND class_id=%s AND date=%s
            """, (active_class['student_id'], active_class['id'], current_date))
            if cursor.fetchone():
                return jsonify({
                    'message': f'✔️ {matched_name} already marked Present',
                    'color': 'blue',
                    'confidence': confidence,
                    'current_class': active_class['subject_name']
                })

            # Mark present
            cursor.execute("""
                INSERT INTO attendance (student_id, date, time, status, class_id, method)
                VALUES (%s, %s, %s, 'Present', %s, 'face')
            """, (active_class['student_id'], current_date, current_time, active_class['id']))
            conn.commit()

            return jsonify({
                'message': f'✅ {matched_name} marked Present!',
                'color': 'green',
                'confidence': confidence,
                'current_class': active_class['subject_name']
            })
        finally:
            cursor.close()
            conn.close()

    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}', 'color': 'red'})

# ==================================================
# 🔄 TOAST NOTIFICATION SYSTEM
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
    try:
        with app.app_context():
            if status == "Present":
                subject_line = f"✅ Khushal College - Present for {subject}"
                body = f"Dear {student_name},\n\nYour attendance has been marked as **PRESENT** at Khushal Degree College:\n\n📚 Subject: {subject}\n📅 Date: {date}\n⏰ Time: {time if time else 'During class hours'}\n\nKeep up the good attendance! 🎉\n\nBest regards,\nFace Attendance System\nKhushal Degree College"
            elif status == "Absent":
                subject_line = f"⚠️ Khushal College - Absent for {subject}"
                body = f"Dear {student_name},\n\nYour attendance has been marked as **ABSENT** at Khushal Degree College:\n\n📚 Subject: {subject}\n📅 Date: {date}\n\nPlease contact your professor if this is incorrect.\n\nBest regards,\nFace Attendance System\nKhushal Degree College"
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
    try:
        with app.app_context():
            if status == "Approved":
                subject_line = f"✅ Khushal College - Leave Approved for {subject}"
                body = f"Dear {student_name},\n\nYour leave application has been **APPROVED** at Khushal Degree College ✅\n\n📚 Subject: {subject}\n🎯 Purpose: {purpose if purpose else 'Leave'}\n📅 Period: {start_date} to {end_date}\n\nYour attendance will be marked as 'Leave' for this period.\n\nBest regards,\nFace Attendance System\nKhushal Degree College"
            elif status == "Rejected":
                subject_line = f"❌ Khushal College - Leave Rejected for {subject}"
                body = f"Dear {student_name},\n\nYour leave application has been **REJECTED** at Khushal Degree College ❌\n\n📚 Subject: {subject}\n🎯 Purpose: {purpose if purpose else 'Leave'}\n📅 Period: {start_date} to {end_date}\n\nPlease contact college administration if you have questions.\n\nBest regards,\nFace Attendance System\nKhushal Degree College"
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
                print(f"💥 Background Email Error for {email_data['student_email']} - {e}")
        print(f"🎯 Background Email Task Completed: {success_count} successful, {fail_count} failed")
    
    thread = threading.Thread(target=email_worker)
    thread.daemon = True
    thread.start()

# ==================================================
# 🕒 AUTO-ABSENT SCHEDULER
# ==================================================
def mark_absentees_job():
    print("🕒 Scheduler: Checking for ended classes...")
    db = get_db_connection()
    if not db: return

    try:
        cursor = db.cursor(dictionary=True)
        now = datetime.now()
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
        if not ended_classes: return

        for cls in ended_classes:
            class_id = cls['id']
            semester = cls['semester']
            subject = cls['subject_name']
            class_end = cls['end_time']
            
            cursor.execute("""
                SELECT id, name, email FROM students 
                WHERE semester = %s 
                AND status = 'approved'
                AND id NOT IN (SELECT student_id FROM attendance WHERE date = %s AND class_id = %s)
                AND id NOT IN (SELECT student_id FROM leaves WHERE status = 'Approved' AND %s BETWEEN start_date AND end_date)
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
    except Exception as e:
        print(f"💥 Scheduler Error: {e}")
    finally:
        if db: db.close()

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(mark_absentees_job, 'interval', minutes=1)
scheduler.start()

# ==================================================
# 🧪 EMAIL TEST ROUTES
# ==================================================
@app.route('/test_college_email')
def test_college_email():
    try:
        msg = Message(
            subject="🎓 Khushal Degree College - Email System Active!",
            recipients=[os.environ.get('MAIL_USERNAME', 'khushaldegreecollege@gmail.com')],
            body="Your Face Attendance System email notification system is operational!"
        )
        mail.send(msg)
        return "✅ College email system ACTIVATED successfully!"
    except Exception as e:
        return f"❌ Email test failed: {str(e)}"

# ==================================================
# 🏠 MAIN ROUTES
# ==================================================
@app.route('/')
def index():
    if 'role' in session and session['role'] == 'admin':
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'role' in session and session['role'] == 'admin':
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        ADMIN_USERNAME = os.environ.get('ADMIN_USER', 'admin')
        ADMIN_PASSWORD = os.environ.get('ADMIN_PASS', 'admin123')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['role'] = 'admin'
            return redirect(url_for('index'))
        else:
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

        cursor.execute("SELECT COUNT(*) AS total FROM students WHERE status='approved'")
        student_count = cursor.fetchone()
        total_students = student_count['total'] if student_count else 0

        today = datetime.now().date()
        cursor.execute("SELECT COUNT(DISTINCT student_id) AS present_today FROM attendance WHERE date = %s AND status = 'Present'", (today,))
        present_result = cursor.fetchone()
        present_today = present_result['present_today'] if present_result else 0

        now = datetime.now()
        current_time = now.time()
        current_day = now.strftime("%A")
        
        days_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        cursor.execute("SELECT subject_name, day_of_week, start_time FROM classes")
        all_classes = cursor.fetchall()
        
        upcoming_class_name = "No Classes"
        if all_classes and current_day in days_map:
            current_day_idx = days_map[current_day]
            min_diff = None
            upcoming_cls = None
            
            for cls in all_classes:
                day = cls['day_of_week']
                if day not in days_map: continue
                day_idx = days_map[day]
                
                start_t = cls['start_time']
                if isinstance(start_t, timedelta):
                    start_t = (datetime.min + start_t).time()
                elif isinstance(start_t, str):
                    try: start_t = datetime.strptime(start_t, "%H:%M:%S").time()
                    except ValueError:
                        try: start_t = datetime.strptime(start_t, "%H:%M").time()
                        except ValueError: continue
                
                day_diff = (day_idx - current_day_idx) % 7
                if day_diff == 0 and start_t <= current_time: day_diff = 7
                
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
                if day_diff == 0: day_label = "Today" if t > current_time else f"Next {day_str}"
                elif day_diff == 1: day_label = "Tomorrow"
                else: day_label = day_str
                upcoming_class_name = f"{upcoming_cls['subject_name']} ({day_label} {time_str})"

        cursor.execute("SELECT COUNT(*) as count FROM students WHERE status='pending'")
        s_pending = cursor.fetchone()['count']
        cursor.execute("SELECT COUNT(*) as count FROM professors WHERE status='pending'")
        p_pending = cursor.fetchone()['count']
        
        return jsonify({
            "students": total_students,
            "present_today": present_today,
            "upcoming_class": upcoming_class_name,
            "pending_signups": s_pending + p_pending 
        })
    except Exception as e:
        print("❌ Dashboard stats error:", e)
        return jsonify({"students": 0, "present_today": 0, "upcoming_class": "Error", "pending_signups": 0})
    finally:
        if cursor: cursor.close()
        if db: db.close()

# ==================================================
# 🎓 STUDENT MANAGEMENT
# ==================================================
@app.route('/manage_students')
@admin_required
def manage_students():
    sem_filter = request.args.get('semester')
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    sql = "SELECT id, name, roll_no, email, semester FROM students WHERE status='approved'"
    params = []
    if sem_filter and sem_filter != "All":
        sql += " AND semester = %s"
        params.append(sem_filter)
    sql += " ORDER BY roll_no"
    cursor.execute(sql, tuple(params))
    students = cursor.fetchall()
    db.close()
    return render_template('manage_students.html', students=students, selected_semester=sem_filter)

@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@admin_required
def edit_student(student_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        name = request.form['name']
        roll_no = request.form['roll_no']
        email = request.form['email']
        semester = request.form.get('semester', '1st Semester')
        cursor.execute("UPDATE students SET name=%s, roll_no=%s, email=%s, semester=%s WHERE id=%s", 
                      (name, roll_no, email, semester, student_id))
        db.commit()
        db.close()
        return redirect(url_for('manage_students'))
    
    cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
    student = cursor.fetchone()
    db.close()
    if not student: return "Student not found", 404
    return render_template('edit_student.html', student=student)

@app.route('/delete_student/<int:student_id>')
@admin_required
def delete_student(student_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT roll_no, name FROM students WHERE id=%s", (student_id,))
    student = cursor.fetchone()
    if student:
        cursor.execute("DELETE FROM students WHERE id=%s", (student_id,))
        db.commit()
        folder_name = f"{student['roll_no']}_{student['name'].replace(' ', '_')}"
        face_path = os.path.join(FACES_DIR, folder_name)
        if os.path.exists(face_path):
            import shutil
            shutil.rmtree(face_path)
            print(f"🗑 Deleted folder: {face_path}")
        reload_faces_background()
    db.close()
    return redirect(url_for('manage_students'))

# ==================================================
# 🎓 STUDENT AUTHENTICATION & SIGNUP
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
            
            if password != confirm_password:
                return render_template('student_signup.html', error="Passwords do not match!")
            
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            
            cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
            if cursor.fetchone():
                return render_template('student_signup.html', error="Roll Number already exists!")

            safe_name = name.replace(" ", "_")
            student_folder = os.path.join(FACES_DIR, f"{roll_no}_{safe_name}")
            os.makedirs(student_folder, exist_ok=True)

            # Save 15 KYC samples from frontend array
            samples_json = request.form.get('face_samples', '[]')
            samples = json.loads(samples_json)
            
            if not samples:
                return render_template('student_signup.html', error="Please complete face verification.")

            main_image_path = ""
            for i, sample_b64 in enumerate(samples):
                try:
                    header, encoded = sample_b64.split(',', 1) if ',' in sample_b64 else ('', sample_b64)
                    img_bytes = base64.b64decode(encoded)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        sample_path = os.path.join(student_folder, f'sample_{i:02d}.jpg')
                        cv2.imwrite(sample_path, img)
                        if i == 0:
                            main_image_path = sample_path
                except: pass

            if not main_image_path:
                main_image_path = os.path.join(student_folder, "sample_00.jpg")

            hashed_pw = generate_password_hash(password)
            
            cursor.execute("INSERT INTO students (name, roll_no, email, password, semester, image_path, status) VALUES (%s, %s, %s, %s, %s, %s, 'pending')", 
                           (name, roll_no, email, hashed_pw, semester, main_image_path))
            db.commit()
            
            reload_faces_background()
            return render_template('student_signup.html', message="✅ Registration Successful!")

        except Exception as e:
            print(f"Error: {e}")
            return render_template('student_signup.html', error=f"Error: {e}")
        finally:
            if db: db.close()

    return render_template('student_signup.html')

@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if 'role' in session and session['role'] == 'student':
        return redirect(url_for('student_dashboard'))
        
    if request.method == 'POST':
        db = get_db_connection()
        if not db: return render_template('student_login.html', error="Database connection error")
        cursor = db.cursor(dictionary=True)

        roll_no = request.form['roll_no']
        password = request.form.get('password', '')
        cursor.execute("SELECT * FROM students WHERE roll_no = %s", (roll_no,))
        student = cursor.fetchone()
        db.close()
        
        if student:
            if student['password']:
                if check_password_hash(student['password'], password):
                    if student['status'] == 'approved':
                        session['logged_in'] = True
                        session['role'] = 'student'
                        session['user_id'] = student['id']
                        session['name'] = student['name']
                        return redirect(url_for('student_dashboard'))
                    elif student['status'] == 'pending':
                        return render_template('student_login.html', error="⏳ Account pending approval")
                    else:
                        return render_template('student_login.html', error="❌ Account rejected")
                else:
                    return render_template('student_login.html', error="Invalid password")
            else:
                session['logged_in'] = True
                session['role'] = 'student'
                session['user_id'] = student['id']
                session['name'] = student['name']
                return redirect(url_for('student_dashboard'))
        else:
            return render_template('student_login.html', error="Student not found")
                
    return render_template('student_login.html')

@app.route('/student_dashboard')
def student_dashboard():
    if 'role' not in session or session['role'] != 'student':
        return redirect(url_for('student_login'))
    
    student_id = session['user_id']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT c.subject_name, COUNT(a.id) as total_classes,
               SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as presents,
               ROUND(SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) * 100.0 / COUNT(a.id), 1) as percentage
        FROM attendance a JOIN classes c ON a.class_id = c.id
        WHERE a.student_id = %s GROUP BY c.subject_name
    """, (student_id,))
    attendance_data = cursor.fetchall()
    
    cursor.execute("SELECT * FROM leaves WHERE student_id = %s ORDER BY created_at DESC", (student_id,))
    leave_requests = cursor.fetchall()
    db.close()
    
    grouped_leaves = {}
    for req in leave_requests:
        date_key = req['created_at'].strftime('%A, %B %d, %Y') if req['created_at'] else "Unknown Date"
        if date_key not in grouped_leaves:
            grouped_leaves[date_key] = []
        grouped_leaves[date_key].append(req)
    
    return render_template('student_dashboard.html', attendance_data=attendance_data, grouped_leaves=grouped_leaves, student_name=session['name'])

@app.route('/student_logout')
def student_logout():
    session.clear()
    return redirect(url_for('student_login'))

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
        cursor = db.cursor()
        try:
            cursor.execute("SELECT id FROM professors WHERE email=%s", (email,))
            if cursor.fetchone():
                return render_template('professor_signup.html', error="❌ Email already registered!")
            
            cursor.execute("INSERT INTO professors (name, email, password, status) VALUES (%s, %s, %s, 'pending')", (name, email, hashed_pw))
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
    if 'role' in session and session['role'] == 'professor':
        return redirect(url_for('professor_dashboard'))
        
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM professors WHERE email=%s", (email,))
        prof = cursor.fetchone()
        db.close()

        if prof:
            if prof['status'] != 'approved':
                return render_template('professor_login.html', error="⏳ Account pending approval.")
            if check_password_hash(prof['password'], password) or prof['password'] == password:
                session['logged_in'] = True
                session['role'] = 'professor'
                session['user_id'] = prof['id']
                session['name'] = prof['name']
                return redirect(url_for('professor_dashboard'))
        
        return render_template('professor_login.html', error="❌ Invalid credentials")
    return render_template('professor_login.html')

@app.route('/manage_professors')
@admin_required
def manage_professors():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM professors WHERE status='approved' ORDER BY name")
    professors = cursor.fetchall()
    db.close()
    return render_template('manage_professors.html', professors=professors)

@app.route('/edit_professor/<int:id>', methods=['GET', 'POST'])
@admin_required
def edit_professor(id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        cursor.execute("UPDATE professors SET name=%s, email=%s WHERE id=%s", (name, email, id))
        db.commit()
        db.close()
        return redirect(url_for('manage_professors'))

    cursor.execute("SELECT * FROM professors WHERE id=%s", (id,))
    professor = cursor.fetchone()
    db.close()
    if not professor: return "Professor not found", 404
    return render_template('edit_professor.html', professor=professor)

@app.route('/delete_professor/<int:id>')
@admin_required
def delete_professor(id):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("DELETE FROM professors WHERE id=%s", (id,))
    db.commit()
    db.close()
    return redirect(url_for('manage_professors'))

# ==================================================
# 🏫 CLASS MANAGEMENT
# ==================================================
@app.route('/edit_class/<int:class_id>', methods=['GET', 'POST'])
@admin_required
def edit_class(class_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        subject = request.form['subject_name']
        prof_id = request.form['professor_id']
        semester = request.form['semester']
        day = request.form['day_of_week']
        start = request.form['start_time']
        end = request.form['end_time']
        cursor.execute("""
            UPDATE classes SET subject_name=%s, professor_id=%s, semester=%s, day_of_week=%s, start_time=%s, end_time=%s WHERE id=%s
        """, (subject, prof_id, semester, day, start, end, class_id))
        db.commit()
        db.close()
        return redirect(url_for('manage_classes'))
    
    cursor.execute("SELECT * FROM classes WHERE id=%s", (class_id,))
    class_info = cursor.fetchone()
    cursor.execute("SELECT id, name FROM professors WHERE status='approved'")
    professors = cursor.fetchall()
    db.close()
    if not class_info: return "Class not found", 404
    
    def format_time(t):
        if hasattr(t, 'seconds'):
            return f"{t.seconds // 3600:02}:{(t.seconds % 3600) // 60:02}"
        return str(t)
    class_info['start_time'] = format_time(class_info['start_time'])
    class_info['end_time'] = format_time(class_info['end_time'])
    return render_template('edit_class.html', class_info=class_info, professors=professors)

@app.route('/manage_classes', methods=['GET', 'POST'])
@admin_required
def manage_classes():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        subject_name = request.form['subject_name']
        professor_id = request.form['professor_id']
        semester = request.form.get('semester', '1st Semester')
        day_of_week = request.form['day_of_week']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        cursor.execute("INSERT INTO classes (subject_name, professor_id, semester, day_of_week, start_time, end_time) VALUES (%s, %s, %s, %s, %s, %s)", 
                      (subject_name, professor_id, semester, day_of_week, start_time, end_time))
        db.commit()
        db.close()
        return redirect(url_for('manage_classes'))
    
    cursor.execute("SELECT id, name FROM professors WHERE status='approved'")
    professors = cursor.fetchall()
    cursor.execute("""
        SELECT c.id, c.subject_name, c.semester, p.name AS professor_name, c.day_of_week, c.start_time, c.end_time
        FROM classes c LEFT JOIN professors p ON c.professor_id = p.id
        ORDER BY FIELD(c.day_of_week, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
    """)
    classes = cursor.fetchall()
    db.close()
    return render_template('manage_classes.html', professors=professors, classes=classes)

@app.route('/delete_class/<int:class_id>')
@admin_required
def delete_class(class_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("DELETE FROM classes WHERE id=%s", (class_id,))
    db.commit()
    db.close()
    return redirect(url_for('manage_classes'))

# ==================================================
# 📊 ATTENDANCE SUMMARY & LOGS
# ==================================================
@app.route('/view_attendance')
@admin_required
def view_attendance():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT DISTINCT subject_name FROM classes")
    classes = cursor.fetchall()
    db.close()
    return render_template('view_attendance.html', classes=classes)

@app.route('/attendance_summary_v2')
def attendance_summary_v2():
    subject = request.args.get('subject', 'all')
    period = request.args.get('period', 'day')
    semester = request.args.get('semester', 'all')
    
    db = get_db_connection()
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
        row['percentage'] = round((row['presents'] / row['total_classes']) * 100, 1) if row['total_classes'] > 0 else 0
    db.close()
    return jsonify(data)

@app.route('/get_weekly_attendance')
@admin_required
def get_weekly_attendance():
    semester = request.args.get('semester')
    subject = request.args.get('subject')
    start_date_str = request.args.get('start_date') 
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = start_date + timedelta(days=6)
    
    cursor.execute("SELECT id, name, roll_no FROM students WHERE semester = %s ORDER BY roll_no", (semester,))
    students = cursor.fetchall()
    
    query = "SELECT student_id, date, status, time, c.subject_name FROM attendance a JOIN classes c ON a.class_id = c.id WHERE date BETWEEN %s AND %s"
    params = [start_date, end_date]
    if subject != 'all':
        query += " AND c.subject_name = %s"
        params.append(subject)
    cursor.execute(query, tuple(params))
    logs = cursor.fetchall()
    
    stats_query = "SELECT student_id, COUNT(CASE WHEN status='Present' THEN 1 END) as p, COUNT(*) as t FROM attendance a JOIN classes c ON a.class_id = c.id WHERE c.semester = %s"
    stats_params = [semester]
    if subject != 'all':
        stats_query += " AND c.subject_name = %s"
        stats_params.append(subject)
    stats_query += " GROUP BY student_id"
    cursor.execute(stats_query, tuple(stats_params))
    stats_data = {row['student_id']: row for row in cursor.fetchall()}

    attendance_map = {}
    for log in logs:
        sid, date_key = log['student_id'], str(log['date'])
        if sid not in attendance_map: attendance_map[sid] = {}
        attendance_map[sid][date_key] = {'status': log['status'], 'time': str(log['time'])}

    final_data = []
    for s in students:
        stat = stats_data.get(s['id'], {'p': 0, 't': 0})
        final_data.append({
            'name': s['name'], 'roll': s['roll_no'], 'week_data': attendance_map.get(s['id'], {}),
            'overall_percent': round((stat['p'] / stat['t']) * 100) if stat['t'] > 0 else 0
        })
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
    cursor = db.cursor(dictionary=True)
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = start_date + timedelta(days=6)
    
    cursor.execute("SELECT id, name, roll_no FROM students WHERE semester = %s ORDER BY roll_no", (semester,))
    students = cursor.fetchall()
    
    query = "SELECT a.student_id, a.date, a.status, a.time, c.subject_name FROM attendance a JOIN classes c ON a.class_id = c.id WHERE c.professor_id = %s AND a.date BETWEEN %s AND %s"
    params = [professor_id, start_date, end_date]
    if subject != 'all':
        query += " AND c.subject_name = %s"
        params.append(subject)
    cursor.execute(query, tuple(params))
    logs = cursor.fetchall()
    
    stats_query = "SELECT student_id, COUNT(CASE WHEN status='Present' THEN 1 END) as p, COUNT(*) as t FROM attendance a JOIN classes c ON a.class_id = c.id WHERE c.professor_id = %s AND c.semester = %s"
    stats_params = [professor_id, semester]
    if subject != 'all':
        stats_query += " AND c.subject_name = %s"
        stats_params.append(subject)
    stats_query += " GROUP BY student_id"
    cursor.execute(stats_query, tuple(stats_params))
    stats_data = {row['student_id']: row for row in cursor.fetchall()}

    attendance_map = {}
    for log in logs:
        sid, date_key = log['student_id'], str(log['date'])
        if sid not in attendance_map: attendance_map[sid] = {}
        attendance_map[sid][date_key] = {'status': log['status'], 'time': str(log['time'])}

    final_data = []
    for s in students:
        stat = stats_data.get(s['id'], {'p': 0, 't': 0})
        final_data.append({
            'name': s['name'], 'roll': s['roll_no'], 'week_data': attendance_map.get(s['id'], {}),
            'overall_percent': round((stat['p'] / stat['t']) * 100) if stat['t'] > 0 else 0
        })
    db.close()
    return jsonify(final_data)

# ==================================================
# 🎥 LIVE ATTENDANCE PAGE (FRONTEND ENTRY)
# ==================================================
@app.route('/live_attendance')
def live_attendance():
    if 'role' not in session or session['role'] not in ['admin', 'professor']:
        return redirect(url_for('login'))
    return render_template('live_attendance.html')

# ==================================================
# 📝 LEAVE MANAGEMENT
# ==================================================
@app.route('/apply_leave', methods=['GET', 'POST'])
def apply_leave():
    if 'role' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    
    logged_in_student_id = session['user_id']
    db = get_db_connection()
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
            cursor.execute("INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')", (logged_in_student_id, subject_name, application_purpose, application_text, start_date, end_date))
        else:
            cursor.execute("SELECT semester FROM students WHERE id = %s", (logged_in_student_id,))
            student_data_row = cursor.fetchone()
            if student_data_row:
                cursor.execute("SELECT DISTINCT subject_name FROM classes WHERE semester = %s", (student_data_row['semester'],))
                semester_subjects = cursor.fetchall()
                if semester_subjects:
                    for sub in semester_subjects:
                        cursor.execute("INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')", (logged_in_student_id, sub['subject_name'], application_purpose, application_text, start_date, end_date))
                else:
                    cursor.execute("INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')", (logged_in_student_id, None, application_purpose, application_text, start_date, end_date))
            else:
                cursor.execute("INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')", (logged_in_student_id, None, application_purpose, application_text, start_date, end_date))
                
        db.commit()

        if student and student['email']:
            def send_leave_email_async(app_context, student_email, student_name, subject_name, purpose, start, end):
                with app_context:
                    try:
                        msg = Message(subject="📝 Leave Application Submitted", recipients=[student_email], body=f"Dear {student_name},\n\nYour leave application has been submitted successfully.\n\n📚 Subject: {subject_name or 'All Subjects'}\n🎯 Purpose: {purpose}\n📅 Period: {start} to {end}\n\nBest regards,\nKhushal Degree College")
                        mail.send(msg)
                    except Exception as e: print(f"❌ Failed to send email: {e}")
            threading.Thread(target=send_leave_email_async, args=(app.app_context(), student['email'], student['name'], subject_name, application_purpose, start_date, end_date), daemon=True).start()

        db.close()
        return render_template('apply_leave.html', student=student, classes=classes, message="✅ Leave application submitted successfully!")
    
    db.close()
    return render_template('apply_leave.html', student=student, classes=classes)

@app.route('/view_requests', methods=['GET', 'POST']) 
@admin_required
def view_requests():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        req_type = request.form.get('type')
        action = request.form.get('action') 
        
        if req_type == 'student':
            sid = request.form.get('student_id')
            status = 'approved' if action == 'approve' else 'rejected'
            if action == 'reject': cursor.execute("DELETE FROM students WHERE id=%s", (sid,))
            else: 
                cursor.execute("UPDATE students SET status=%s WHERE id=%s", (status, sid))
                reload_faces_background()
                
        elif req_type == 'professor':
            pid = request.form.get('professor_id')
            status = 'approved' if action == 'approve' else 'rejected'
            if action == 'reject': cursor.execute("DELETE FROM professors WHERE id=%s", (pid,))
            else: cursor.execute("UPDATE professors SET status=%s WHERE id=%s", (status, pid))

        db.commit()
        db.close()
        return redirect(url_for('view_requests'))

    cursor.execute("SELECT * FROM students WHERE status='pending'")
    pending_students = cursor.fetchall()
    cursor.execute("SELECT * FROM professors WHERE status='pending'")
    pending_professors = cursor.fetchall()
    db.close()
    return render_template('view_requests.html', pending_students=pending_students, pending_professors=pending_professors)

@app.route('/professor_leaves', methods=['GET', 'POST'])
@professor_required
def professor_leaves():
    professor_id = session['user_id']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        leave_id = request.form.get('leave_id')
        action = request.form.get('action') 
        cursor.execute("UPDATE leaves SET status=%s WHERE id=%s", (action, leave_id))
        db.commit()
        flash(f"Leave {action} successfully!", "success")
        return redirect(url_for('professor_leaves'))

    cursor.execute("""
        SELECT DISTINCT l.*, s.name, s.roll_no, s.semester 
        FROM leaves l JOIN students s ON l.student_id = s.id
        WHERE l.status = 'Pending'
        AND (
            l.subject_name IN (SELECT subject_name FROM classes WHERE professor_id = %s)
            OR ((l.subject_name IS NULL OR l.subject_name = '') AND s.semester IN (SELECT semester FROM classes WHERE professor_id = %s))
        ) ORDER BY l.start_date DESC
    """, (professor_id, professor_id))
    
    leave_records = cursor.fetchall()
    db.close()
    return render_template('professor_leaves.html', leave_records=leave_records)

# ==================================================
# 📋 MANUAL ATTENDANCE
# ==================================================
@app.route('/manual_attendance')
def manual_attendance():
    role = session.get('role')
    if role not in ['admin', 'professor']:
        return redirect(url_for('login'))
        
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if role == 'professor': cursor.execute("SELECT * FROM classes WHERE professor_id=%s ORDER BY day_of_week", (session['user_id'],))
    else: cursor.execute("SELECT * FROM classes ORDER BY day_of_week")
    classes = cursor.fetchall()
    db.close()
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('manual_attendance.html', classes=classes, today=today)

@app.route('/get_class_students/<int:class_id>')
def get_class_students(class_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT semester FROM classes WHERE id=%s", (class_id,))
    cls = cursor.fetchone()
    if not cls: return jsonify({'students': []})
    
    cursor.execute("SELECT id, name, roll_no FROM students WHERE semester=%s AND status='approved' ORDER BY name", (cls['semester'],))
    students = cursor.fetchall()
    
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    cursor.execute("SELECT student_id, status FROM attendance WHERE class_id=%s AND date=%s", (class_id, date))
    attendance = {row['student_id']: row['status'] for row in cursor.fetchall()}
    db.close()
    return jsonify({'students': students, 'existing_attendance': attendance})

@app.route('/save_manual_attendance', methods=['POST'])
def save_manual_attendance():
    data = request.json
    class_id, date, attendance_data = data['class_id'], data['date'], data['attendance']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT start_time, subject_name FROM classes WHERE id = %s", (class_id,))
    class_info = cursor.fetchone()
    class_time, subject_name = class_info['start_time'], class_info['subject_name']
    
    for student_id, status in attendance_data.items():
        cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, class_id, date))
        existing = cursor.fetchone()
        if existing: cursor.execute("UPDATE attendance SET status = %s, time = %s, method = 'manual' WHERE id = %s", (status, class_time, existing['id']))
        else: cursor.execute("INSERT INTO attendance (student_id, class_id, date, time, status, method) VALUES (%s, %s, %s, %s, %s, 'manual')", (student_id, class_id, date, class_time, status))
    
    db.commit()
    
    email_data_list = []
    for student_id, status in attendance_data.items():
        if status.lower() in ["present", "absent"]:
            cursor.execute("SELECT name, email FROM students WHERE id = %s", (student_id,))
            student_data = cursor.fetchone()
            if student_data and student_data['email']:
                email_data_list.append({'student_email': student_data['email'], 'student_name': student_data['name'], 'status': status.capitalize(), 'subject': subject_name, 'date': date, 'time': class_time})
    
    db.close()
    if email_data_list: send_attendance_emails_in_background(email_data_list)
    return jsonify({'success': True, 'message': f'Attendance saved for {len(attendance_data)} students.'})

@app.route('/bulk_attendance_action', methods=['POST'])
def bulk_attendance_action():
    data = request.json
    action, student_ids, class_id, date = data['action'], data['student_ids'], data['class_id'], data['date']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT subject_name FROM classes WHERE id = %s", (class_id,))
    subject_name = cursor.fetchone()['subject_name']
    current_time = datetime.now().strftime('%H:%M:%S')
    
    for student_id in student_ids:
        cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, class_id, date))
        existing = cursor.fetchone()
        if existing: cursor.execute("UPDATE attendance SET status = %s WHERE id = %s", (action, existing['id']))
        else: cursor.execute("INSERT INTO attendance (student_id, class_id, date, time, status, method) VALUES (%s, %s, %s, %s, %s, 'manual')", (student_id, class_id, date, current_time, action))
    db.commit()
    
    if action.lower() in ["present", "absent"]:
        email_data_list = []
        for student_id in student_ids:
            cursor.execute("SELECT name, email FROM students WHERE id = %s", (student_id,))
            student_data = cursor.fetchone()
            if student_data and student_data['email']:
                email_data_list.append({'student_email': student_data['email'], 'student_name': student_data['name'], 'status': action.capitalize(), 'subject': subject_name, 'date': date, 'time': current_time})
        if email_data_list: send_attendance_emails_in_background(email_data_list)
    
    db.close()
    return jsonify({'success': True, 'message': f'Bulk {action} applied to {len(student_ids)} students.'})

# ==================================================
# 📁 UTILITY ROUTES
# ==================================================
@app.route('/face_images/<path:filename>')
def face_images(filename):
    return send_from_directory(FACES_DIR, filename)

# ==================================================
# 👨‍🏫 PROFESSOR SPECIFIC ROUTES
# ==================================================
@app.route('/professor_set_password', methods=['GET', 'POST'])
def professor_set_password():
    if request.method == 'GET':
        professor_id, email = request.args.get('professor_id'), request.args.get('email')
        if not professor_id or not email: return redirect(url_for('professor_login'))
        return render_template('professor_set_password.html', professor_id=professor_id, email=email)
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    professor_id, password, confirm_password, email = request.form['professor_id'], request.form['password'], request.form['confirm_password'], request.form['email']
    
    if password != confirm_password: return render_template('professor_set_password.html', error="Passwords do not match!", professor_id=professor_id, email=email)
    
    hashed_password = generate_password_hash(password)
    cursor.execute("UPDATE professors SET password = %s WHERE id = %s", (hashed_password, professor_id))
    db.commit()
    
    cursor.execute("SELECT * FROM professors WHERE id = %s", (professor_id,))
    professor = cursor.fetchone()
    db.close()
    
    session['logged_in'] = True
    session['role'] = 'professor'
    session['user_id'] = professor['id']
    session['name'] = professor['name']
    return redirect(url_for('professor_dashboard'))

@app.route('/professor_dashboard')
@professor_required
def professor_dashboard():
    professor_id = session['user_id']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT name, email FROM professors WHERE id = %s", (professor_id,))
    prof_data = cursor.fetchone()

    today_name = datetime.now().strftime("%A")
    cursor.execute("SELECT * FROM classes WHERE professor_id = %s AND day_of_week = %s ORDER BY start_time ASC", (professor_id, today_name))
    todays_classes = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) as count FROM attendance a JOIN classes c ON a.class_id = c.id WHERE c.professor_id = %s AND a.date = CURDATE() AND a.status = 'Present'", (professor_id,))
    present_count = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(DISTINCT l.id) as count FROM leaves l JOIN students s ON l.student_id = s.id WHERE l.status = 'Pending' AND (l.subject_name IN (SELECT subject_name FROM classes WHERE professor_id = %s) OR ((l.subject_name IS NULL OR l.subject_name = '') AND s.semester IN (SELECT semester FROM classes WHERE professor_id = %s)))", (professor_id, professor_id))
    leaves_count = cursor.fetchone()['count']

    cursor.execute("SELECT * FROM classes WHERE professor_id = %s", (professor_id,))
    prof_all_classes = cursor.fetchall()
    
    next_class = None
    if prof_all_classes:
        min_diff = None
        current_time, current_day = datetime.now().time(), datetime.now().strftime("%A")
        days_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        
        if current_day in days_map:
            current_day_idx = days_map[current_day]
            for cls in prof_all_classes:
                day = cls['day_of_week']
                if day not in days_map: continue
                day_idx = days_map[day]
                
                start_t = cls['start_time']
                if isinstance(start_t, timedelta): start_t = (datetime.min + start_t).time()
                elif isinstance(start_t, str):
                    try: start_t = datetime.strptime(start_t, "%H:%M:%S").time()
                    except ValueError:
                        try: start_t = datetime.strptime(start_t, "%H:%M").time()
                        except ValueError: continue
                
                day_diff = (day_idx - current_day_idx) % 7
                if day_diff == 0 and start_t <= current_time: day_diff = 7
                
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
                if day_diff == 0: day_label = "Today" if t > current_time else f"Next {day_str}"
                elif day_diff == 1: day_label = "Tomorrow"
                else: day_label = day_str
                next_class['display_time'] = f"{day_label} at {time_str}"

    db.close()
    return render_template('professor_dashboard.html', professor=prof_data, classes=todays_classes, present_count=present_count, leaves_count=leaves_count, next_class=next_class)

@app.route('/professor_logout')
def professor_logout():
    session.clear()
    return redirect(url_for('professor_login'))

@app.route('/professor_attendance')
@professor_required
def professor_attendance():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT DISTINCT subject_name FROM classes WHERE professor_id = %s", (session.get('user_id'),))
    professor_subjects = cursor.fetchall()
    db.close()
    return render_template('professor_attendance.html', subjects=professor_subjects)

@app.route('/professor_attendance_summary')
@professor_required
def professor_attendance_summary():
    subject, semester, professor_id = request.args.get('subject', 'all'), request.args.get('semester', 'all'), session['user_id']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    query = "SELECT s.name, s.roll_no, s.semester, c.subject_name, COUNT(CASE WHEN a.status='Present' THEN 1 END) as presents, COUNT(CASE WHEN a.status='Absent' THEN 1 END) as absents, COUNT(CASE WHEN a.status='Leave' THEN 1 END) as leaves, COUNT(a.id) as total_classes FROM students s JOIN attendance a ON s.id = a.student_id JOIN classes c ON a.class_id = c.id WHERE c.professor_id = %s"
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
        row['percentage'] = round((row['presents'] / row['total_classes']) * 100, 1) if row['total_classes'] > 0 else 0
    db.close()
    return jsonify(data)

@app.route('/professor_manual_attendance')
@professor_required
def professor_manual_attendance():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM classes WHERE professor_id = %s ORDER BY subject_name", (session.get('user_id'),))
    professor_classes = cursor.fetchall()
    db.close()

    formatted_classes = []
    for cls in professor_classes:
        formatted_class = dict(cls)
        formatted_class['start_time'] = str(formatted_class['start_time'])
        formatted_class['end_time'] = str(formatted_class['end_time'])
        formatted_classes.append(formatted_class)
    
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('professor_manual_attendance.html', classes=formatted_classes, today=today)

@app.route('/professor_approve_leave', methods=['POST'])
@professor_required
def professor_approve_leave():
    leave_id, action = request.form['leave_id'], request.form['action']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT l.*, s.name, s.email, s.semester FROM leaves l JOIN students s ON l.student_id = s.id WHERE l.id = %s", (leave_id,))
        leave = cursor.fetchone()
        if not leave: return jsonify({'success': False, 'error': 'Leave not found'})
        
        cursor.execute("UPDATE leaves SET status = %s WHERE id = %s", (action, leave_id))
        
        if action == 'Approved':
            student_id, subject_name, semester, start_date, end_date = leave['student_id'], leave['subject_name'], leave['semester'], leave['start_date'], leave['end_date']
            if isinstance(start_date, str): start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            if isinstance(end_date, str): end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

            current_date = start_date
            while current_date <= end_date:
                if subject_name: cursor.execute("SELECT id FROM classes WHERE subject_name = %s AND semester = %s", (subject_name, semester))
                else: cursor.execute("SELECT id FROM classes WHERE semester = %s", (semester,))
                
                for cls in cursor.fetchall():
                    cursor.execute("DELETE FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, cls['id'], current_date))
                    cursor.execute("INSERT INTO attendance (student_id, class_id, date, time, status, method) VALUES (%s, %s, %s, NOW(), 'Leave', 'system')", (student_id, cls['id'], current_date))
                current_date += timedelta(days=1)
        db.commit()
        return jsonify({'success': True, 'message': f'Leave {action} and attendance updated!'})
    except Exception as e:
        if db: db.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cursor.close()
        db.close()

# ==================================================
# 🏁 CLEANUP & STARTUP
# ==================================================
@atexit.register
def cleanup_on_exit():
    if scheduler.running:
        scheduler.shutdown()
    print("Application exiting. Scheduler stopped.")

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    )