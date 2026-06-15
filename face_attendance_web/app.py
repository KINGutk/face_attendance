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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for 6 KYC images

# ==================================================
# 🧠 AI MODEL LOADING (YOLOv8 + ResNet50)
# ==================================================
print("🔄 Loading YOLOv8 and ResNet50 models...")

MODEL_PATH = 'yolov8n-face.pt'
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading specialized YOLOv8 Face model from Mirror...")
    urls = [
        "https://github.com/SannketNikam/Face-Detection/raw/main/yolov8n-face.pt",
        "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt"
    ]
    for url in urls:
        try:
            print(f"🔗 Trying: {url}")
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("✅ YOLOv8 downloaded!")
            break
        except:
            pass

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
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_face_embedding(img_bgr):
    """Generates a 512-dimensional math map of the face using PyTorch"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = face_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedding_model(tensor)
    return torch.nn.functional.normalize(emb, dim=1).squeeze().cpu().numpy()

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
    if not db: return

    try:
        cursor = db.cursor(dictionary=True)
        # Ensure face_data column exists
        try:
            cursor.execute("ALTER TABLE students ADD COLUMN face_data LONGTEXT;")
            db.commit()
            print("✅ 'face_data' column added to DB!")
        except: pass
        
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
    return jsonify({"success": True, "message": "Face DB Reloaded"})

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
# ⚡ INSTANT PHOTO CHECK (YOLOv8)
# ==================================================
@app.route('/check_photo_quality', methods=['POST'])
def check_photo_quality():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data: return jsonify({"valid": False, "error": "No image data"})
        
        _, encoded = image_data.split(",", 1) if "," in image_data else ('', image_data)
        np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 1. Blur Check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:
            return jsonify({"valid": False, "error": "⚠️ Too Blurry. Hold steady!"})

        # 2. YOLO Face Detection
        results = yolo_model(img, verbose=False)
        faces = [box for r in results for box in r.boxes if box.conf[0] >= 0.55]
        
        if not faces: return jsonify({"valid": False, "error": "⚠️ No face detected."})
        if len(faces) > 1: return jsonify({"valid": False, "error": "⚠️ Multiple faces!"})

        # 3. Size Check
        box = faces[0]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        face_area = (x2-x1) * (y2-y1)
        image_area = img.shape[0] * img.shape[1]
        if face_area < (image_area * 0.08):
            return jsonify({"valid": False, "error": "⚠️ Move closer."})

        return jsonify({"valid": True})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})

# ==================================================
# ⚡ LIVE ATTENDANCE API (Cosine Similarity)
# ==================================================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data: return jsonify({"message": "No Image", "color": "red"})

        _, encoded = image_data.split(",", 1) if "," in image_data else ('', image_data)
        np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Get Current Class
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        now = datetime.now()
        date_today = now.date()
        time_now = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")
        
        cursor.execute("SELECT * FROM classes WHERE day_of_week=%s AND start_time<=%s AND end_time>=%s LIMIT 1", (day_name, time_now, time_now))
        current_class = cursor.fetchone()
        class_info = f"{current_class['subject_name']} ({current_class['semester']})" if current_class else "No Active Class"

        if not KNOWN_ENCODINGS: return jsonify({"message": "DB Empty", "color": "orange", "current_class": class_info})

        # YOLO Detection
        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        results = yolo_model(small, verbose=False)
        faces = [box for r in results for box in r.boxes if box.conf[0] >= 0.5]
        
        if not faces: return jsonify({"message": "No face detected", "color": "orange", "current_class": class_info})

        best_box = max(faces, key=lambda b: float(b.conf[0]))
        x1,y1,x2,y2 = [v*2 for v in map(int, best_box.xyxy[0])]
        face_crop = img[y1:y2, x1:x2]
        
        if face_crop.size == 0: return jsonify({"message": "Crop error", "color": "red"})

        # PyTorch Embedding & Cosine Similarity Match
        query_emb = get_face_embedding(face_crop)
        sims = [float(np.dot(query_emb, k) / (np.linalg.norm(query_emb)*np.linalg.norm(k)+1e-8)) for k in KNOWN_ENCODINGS]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        THRESHOLD = 0.72  # Strict 99% accuracy threshold
        if best_sim < THRESHOLD:
            return jsonify({"message": "Unknown Face", "color": "red", "current_class": class_info})

        name = KNOWN_NAMES[best_idx]
        roll = KNOWN_ROLLS[best_idx]

        if not current_class:
            return jsonify({"message": f"👤 Recognized: {name} (No Class)", "color": "cyan", "current_class": class_info})

        # Mark Attendance
        cursor.execute("SELECT id FROM students WHERE roll_no=%s", (roll,))
        student = cursor.fetchone()
        if student:
            cursor.execute("SELECT id FROM attendance WHERE student_id=%s AND date=%s AND class_id=%s", (student['id'], date_today, current_class['id']))
            if not cursor.fetchone():
                cursor.execute("INSERT INTO attendance (student_id, date, time, status, class_id, method) VALUES (%s, %s, %s, 'Present', %s, 'auto')", 
                               (student['id'], date_today, time_now, current_class['id']))
                db.commit()
                return jsonify({"message": f"✅ Present: {name}", "color": "green", "current_class": class_info})
            else:
                return jsonify({"message": f"ℹ️ Already Marked: {name}", "color": "blue", "current_class": class_info})
                
        return jsonify({"message": "Student DB Error", "color": "red", "current_class": class_info})

    except Exception as e:
        return jsonify({"message": f"Server Error: {e}", "color": "red", "current_class": "Error"})
    finally:
        if 'cursor' in locals() and cursor: cursor.close()
        if 'db' in locals() and db: db.close()

# ==================================================
# 🎓 STUDENT SIGNUP (JSON DATABASE SAVE)
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
            semester = request.form.get('semester', '6th Semester')

            samples_json = request.form.get('face_samples', '[]')
            samples = json.loads(samples_json)
            
            if len(samples) < 6:
                return render_template('student_signup.html', error="Please complete all 6 face angles.")

            student_encodings = []
            
            for b64_img in samples:
                header, encoded = b64_img.split(',', 1) if ',' in b64_img else ('', b64_img)
                np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    results = yolo_model(img, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            if box.conf[0] >= 0.55:
                                x1,y1,x2,y2 = map(int, box.xyxy[0])
                                face_crop = img[y1:y2, x1:x2]
                                if face_crop.size > 0:
                                    embedding = get_face_embedding(face_crop)
                                    student_encodings.append(embedding.tolist())

            if not student_encodings:
                return render_template('student_signup.html', error="AI could not map your face. Please try again in better light.")

            # Map ko Text (JSON) mein convert karein
            face_data_json = json.dumps(student_encodings)

            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            
            cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
            if cursor.fetchone():
                return render_template('student_signup.html', error="Roll Number already exists!")

            hashed_pw = generate_password_hash(password)
            
            # DB mein insert karein (Bina folder/images save kiye!)
            cursor.execute("""
                INSERT INTO students (name, roll_no, email, password, semester, status, face_data) 
                VALUES (%s, %s, %s, %s, %s, 'pending', %s)
            """, (name, roll_no, email, hashed_pw, semester, face_data_json))
            
            db.commit()
            load_known_faces() # Reload AI Memory
            
            return render_template('student_signup.html', message="✅ Registration Successful! Pending Admin Approval.")

        except Exception as e:
            return render_template('student_signup.html', error=f"Error: {e}")
        finally:
            if db: db.close()

    return render_template('student_signup.html')


# ==================================================
# 🌐 ALL OTHER EXISTING ROUTES (Intact & Cleaned)
# ==================================================

@app.route('/')
def index():
    if session.get('role') == 'admin': return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('role') == 'admin': return redirect(url_for('index'))
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

@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if session.get('role') == 'student': return redirect(url_for('student_dashboard'))
    if request.method == 'POST':
        db = get_db_connection()
        if not db: return render_template('student_login.html', error="DB error")
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE roll_no=%s", (request.form['roll_no'],))
        st = cursor.fetchone()
        db.close()
        
        if st and st['password'] and check_password_hash(st['password'], request.form['password']):
            if st['status'] == 'approved':
                session.update({'logged_in': True, 'role': 'student', 'user_id': st['id'], 'name': st['name']})
                return redirect(url_for('student_dashboard'))
            return render_template('student_login.html', error="Account pending/rejected.")
        return render_template('student_login.html', error="Invalid credentials.")
    return render_template('student_login.html')

@app.route('/student_dashboard')
def student_dashboard():
    if session.get('role') != 'student': return redirect(url_for('student_login'))
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT c.subject_name, COUNT(a.id) as t, SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) as p FROM attendance a JOIN classes c ON a.class_id=c.id WHERE a.student_id=%s GROUP BY c.subject_name", (session['user_id'],))
    att = cursor.fetchall()
    cursor.execute("SELECT * FROM leaves WHERE student_id=%s ORDER BY created_at DESC", (session['user_id'],))
    leaves = cursor.fetchall()
    db.close()
    
    for row in att: row['percentage'] = round((row['p']/row['t'])*100,1) if row['t']>0 else 0
    grouped_leaves = {}
    for r in leaves:
        d = r['created_at'].strftime('%A, %B %d, %Y') if r['created_at'] else "Unknown"
        grouped_leaves.setdefault(d, []).append(r)
        
    return render_template('student_dashboard.html', attendance_data=att, grouped_leaves=grouped_leaves, student_name=session['name'])

@app.route('/student_logout')
def student_logout():
    session.clear()
    return redirect(url_for('student_login'))

@app.route('/live_attendance')
def live_attendance():
    if session.get('role') not in ['admin', 'professor']: return redirect(url_for('login'))
    return render_template('live_attendance.html')

@app.route('/dashboard_stats')
def dashboard_stats():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) as t FROM students WHERE status='approved'")
    total_st = cursor.fetchone()['t']
    cursor.execute("SELECT COUNT(DISTINCT student_id) as p FROM attendance WHERE date=CURDATE() AND status='Present'")
    present_td = cursor.fetchone()['p']
    cursor.execute("SELECT COUNT(*) as count FROM students WHERE status='pending'")
    s_pen = cursor.fetchone()['count']
    cursor.execute("SELECT COUNT(*) as count FROM professors WHERE status='pending'")
    p_pen = cursor.fetchone()['count']
    db.close()
    return jsonify({"students": total_st, "present_today": present_td, "upcoming_class": "Dashboard Active", "pending_signups": s_pen + p_pen})

@app.route('/manage_students')
@admin_required
def manage_students():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM students WHERE status='approved' ORDER BY roll_no")
    st = cursor.fetchall()
    db.close()
    return render_template('manage_students.html', students=st, selected_semester="All")

@app.route('/view_requests', methods=['GET', 'POST'])
@admin_required
def view_requests():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        rtype, action, sid = request.form.get('type'), request.form.get('action'), request.form.get('student_id') or request.form.get('professor_id')
        stat = 'approved' if action == 'approve' else 'rejected'
        tbl = 'students' if rtype == 'student' else 'professors'
        if action == 'reject': cursor.execute(f"DELETE FROM {tbl} WHERE id=%s", (sid,))
        else: cursor.execute(f"UPDATE {tbl} SET status=%s WHERE id=%s", (stat, sid))
        db.commit()
        if rtype == 'student' and action == 'approve': load_known_faces()
        return redirect(url_for('view_requests'))
    cursor.execute("SELECT * FROM students WHERE status='pending'")
    ps = cursor.fetchall()
    cursor.execute("SELECT * FROM professors WHERE status='pending'")
    pp = cursor.fetchall()
    db.close()
    return render_template('view_requests.html', pending_students=ps, pending_professors=pp)

@app.route('/delete_student/<int:student_id>')
@admin_required
def delete_student(student_id):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("DELETE FROM students WHERE id=%s", (student_id,))
    db.commit()
    db.close()
    load_known_faces()
    return redirect(url_for('manage_students'))

# ==================================================
# 🚀 ATE-EXIT & RUN
# ==================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true')