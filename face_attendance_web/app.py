from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session, Response, flash
import mysql.connector
from datetime import datetime, timedelta
import os
import sys

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
import face_recognition
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import threading
import time
import atexit
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler

# ==================================================
# 🌍 LOAD ENVIRONMENT VARIABLES (MUST BE FIRST)
# ==================================================
try:
    from dotenv import load_dotenv
    # Look for .env in the project root (one level up from face_attendance_web/)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        load_dotenv()  # Fallback: search current dir
    print("✅ Environment variables loaded from .env")
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables.")

# ==================================================
# 🔧 FLASK APP CONFIG
# ==================================================

app = Flask(__name__)

# --- Load Config from Environment Variables ---
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super_secure_authentic_key_2026')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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
    """
    Creates a new database connection.
    Supports both local MySQL (XAMPP) and cloud TiDB Serverless.
    """
    try:
        db_host = os.environ.get('DB_HOST', 'localhost')
        db_port = int(os.environ.get('DB_PORT', 3306))
        
        conn_params = {
            'host': db_host,
            'port': db_port,
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASS', ''),
            'database': os.environ.get('DB_NAME', 'face_attendance_db'),
        }
        
        # TiDB Serverless requires SSL — auto-enable for non-localhost
        if db_host != 'localhost' and db_host != '127.0.0.1':
            conn_params['ssl_disabled'] = False
        
        conn = mysql.connector.connect(**conn_params)
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Database connection error: {err}")
        return None

# ==================================================
# 🔑 AUTHENTICATION
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
# 🎥 CAMERA MANAGEMENT
# ==================================================
camera = None
camera_lock = threading.Lock()
camera_active = False

def release_camera():
    """Safely release the camera resources"""
    global camera, camera_active
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                print("✅ Camera released successfully")
            except Exception as e:
                print(f"❌ Error releasing camera: {e}")
            finally:
                camera = None
                camera_active = False
        try:
            cv2.destroyAllWindows()
        except:
            pass

# ==================================================
# 🖼️ FACE RECOGNITION CACHE (UPDATED FOR 3 ANGLES)
# ==================================================

KNOWN_ENCODINGS = []
KNOWN_NAMES = []
KNOWN_ROLLS = []

def load_known_faces():
    """
    Loads ALL angles (Front, Left, Right) for every approved student.
    """
    print("🔄 Loading face database (3-Angle Mode)...")
    global KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS
    KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS = [], [], []
    
    db = get_db_connection()
    if not db:
        print("❌ Database connection failed")
        return

    try:
        cursor = db.cursor(dictionary=True)
        # Get approved students
        cursor.execute("SELECT roll_no, name, image_path FROM students WHERE status = 'approved'")
        students = cursor.fetchall()
        
        for student in students:
            # The DB stores the path to 'front.jpg'. 
            # We need the PARENT FOLDER to find 'left.jpg' and 'right.jpg'.
            if student['image_path']:
                student_folder = os.path.dirname(student['image_path'])
                
                if os.path.exists(student_folder):
                    # Loop through Front, Left, Right images in that folder
                    for filename in os.listdir(student_folder):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(student_folder, filename)
                            
                            try:
                                img = cv2.imread(img_path)
                                if img is None: continue
                                
                                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                encs = face_recognition.face_encodings(rgb)
                                
                                if encs:
                                    # Add this specific angle to memory
                                    KNOWN_ENCODINGS.append(encs[0])
                                    KNOWN_NAMES.append(student['name'])
                                    KNOWN_ROLLS.append(student['roll_no'])
                                    # print(f"   ✅ Loaded: {student['name']} ({filename})")
                                    
                            except Exception as e:
                                print(f"⚠️ Error loading {filename} for {student['name']}: {e}")
                else:
                    print(f"⚠️ Folder not found for {student['name']}")

        print(f"✅ Loaded {len(KNOWN_ENCODINGS)} total face angles.")
        
    except Exception as e:
        print(f"❌ Error loading faces: {e}")
    finally:
        db.close()

# Load faces on initial startup
load_known_faces()

@app.route('/reload_faces')
# @admin_required  <-- Uncomment if you have this decorator
def reload_faces():
    load_known_faces()
    return jsonify({"success": True, "message": f"Face cache reloaded. {len(KNOWN_ENCODINGS)} angles loaded."})

# ==================================================
# 🛡️ STRICTER SECURITY CHECKS
# ==================================================


def is_image_blurry(image_bytes, threshold=150):
    """
    Checks blurriness.
    Threshold increased to 150 (Stricter).
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: return True, 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"🔎 Blur Score: {int(score)} (Threshold: {threshold})") # Debug Print
        return score < threshold, score
    except Exception as e:
        print(f"⚠️ Blur check error: {e}")
        return True, 0

def validate_three_angles(front_bytes, left_bytes, right_bytes):
    """
    Strict validation:
    1. Face must be detected.
    2. Face must be LARGE (at least 10% of image) to avoid 'Car' false positives.
    3. Faces must match identity.
    """
    try:
        def get_face_data(img_bytes, label):
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: return None, f"Could not decode {label} image"
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 1. Detect Face Boxes
            boxes = face_recognition.face_locations(rgb)
            if not boxes: 
                return None, f"No face detected in {label} photo."
            
            # 2. Check Face Size (Reject Tiny False Positives like cars)
            top, right, bottom, left = boxes[0]
            face_area = (bottom - top) * (right - left)
            h, w, _ = img.shape
            image_area = h * w
            
            if face_area < (image_area * 0.08): 
                print(f"⚠️ Rejected {label}: Face too small ({int((face_area/image_area)*100)}%)")
                return None, f"Face in {label} photo is too small. Please move closer."

            # 3. Get Encoding
            encs = face_recognition.face_encodings(rgb, boxes)
            return encs[0], None

        # Process all 3 images
        enc_front, err_front = get_face_data(front_bytes, "FRONT")
        # 🟢 FIX: Check for None explicitly
        if enc_front is None: return False, err_front
        
        enc_left, err_left = get_face_data(left_bytes, "LEFT")
        if enc_left is None: return False, err_left
        
        enc_right, err_right = get_face_data(right_bytes, "RIGHT")
        if enc_right is None: return False, err_right

        # 4. Verify Identity (Cousin Defense)
        match_left = face_recognition.compare_faces([enc_front], enc_left, tolerance=0.5)[0]
        match_right = face_recognition.compare_faces([enc_front], enc_right, tolerance=0.5)[0]

        if not match_left: return False, "Left profile does not match Front face!"
        if not match_right: return False, "Right profile does not match Front face!"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation Error: {str(e)}"
    
    # ==================================================
# ⚡ LIVE ATTENDANCE API (Connects to Frontend)
# ==================================================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"message": "No Image", "color": "red", "current_class": "--"})

        # --- 1. Decode Image ---
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
        
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # --- 2. Get Current Class ---
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        
        now = datetime.now()
        date_today = now.date()
        time_now = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")
        
        cursor.execute("""
            SELECT * FROM classes 
            WHERE day_of_week = %s AND start_time <= %s AND end_time >= %s 
            LIMIT 1
        """, (day_name, time_now, time_now))
        
        current_class = cursor.fetchone()
        class_info_str = f"{current_class['subject_name']} ({current_class['semester']})" if current_class else "No Active Class"
        
        # --- 3. Debugging Prints ---
        # print(f"📸 Frame received. Known Faces: {len(KNOWN_ENCODINGS)}") # Uncomment to see in terminal

        if len(KNOWN_ENCODINGS) == 0:
            return jsonify({
                "message": "⚠️ System Empty (0 Students Loaded)", 
                "color": "orange", 
                "current_class": class_info_str
            })

        # --- 4. Face Recognition ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, faces)
        
        response_msg = "Scanning..."
        response_color = "white"

        if not faces:
            response_msg = "No Face Detected"
            response_color = "red"
        
        # Check against known faces
        for encoding in encodings:
            matches = face_recognition.compare_faces(KNOWN_ENCODINGS, encoding)
            distances = face_recognition.face_distance(KNOWN_ENCODINGS, encoding)
            
            if len(distances) > 0:
                best_idx = np.argmin(distances)
                
                # Debug Print
                # print(f"🔍 Face Distance: {distances[best_idx]} (Match: {matches[best_idx]})")

                if matches[best_idx] and distances[best_idx] < 0.45: # Strict tolerance
                    name = KNOWN_NAMES[best_idx]
                    roll = KNOWN_ROLLS[best_idx]
                    
                    if current_class:
                        cursor.execute("SELECT id FROM students WHERE roll_no=%s", (roll,))
                        student = cursor.fetchone()
                        
                        if student:
                            # Check if already marked
                            cursor.execute("""
                                SELECT id FROM attendance 
                                WHERE student_id=%s AND date=%s AND class_id=%s
                            """, (student['id'], date_today, current_class['id']))
                            
                            if not cursor.fetchone():
                                cursor.execute("""
                                    INSERT INTO attendance (student_id, date, time, status, class_id, method) 
                                    VALUES (%s, %s, %s, 'Present', %s, 'auto')
                                """, (student['id'], date_today, time_now, current_class['id']))
                                db.commit()
                                
                                response_msg = f"✅ Present: {name}"
                                response_color = "#00ff88"
                                print(f"✅ Marked Present: {name}")
                            else:
                                response_msg = f"ℹ️ Already Marked: {name}"
                                response_color = "cyan"
                        else:
                            response_msg = "Student Not Found in DB"
                            response_color = "red"
                    else:
                        response_msg = f"👤 Recognized: {name} (No Class)"
                        response_color = "yellow"
                else:
                    response_msg = "Unknown Face"
                    response_color = "red"
            else:
                # This handles the empty DB case inside the loop (just in case)
                response_msg = "Unknown (DB Empty)"
                response_color = "orange"

        return jsonify({
            "message": response_msg,
            "color": response_color,
            "current_class": class_info_str
        })

    except Exception as e:
        print(f"❌ Error in process_frame: {e}")
        return jsonify({"message": "Server Error", "color": "red", "current_class": "Error"})
    finally:
        if 'cursor' in locals() and cursor: cursor.close()
        if 'db' in locals() and db: db.close()
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
    """Send attendance notification to student"""
    try:
        # Create application context
        with app.app_context():
            if status == "Present":
                subject_line = f"✅ Khushal College - Present for {subject}"
                body = f"""
Dear {student_name},

Your attendance has been marked as **PRESENT** at Khushal Degree College:

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

Your attendance has been marked as **ABSENT** at Khushal Degree College:

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
        # Create application context
        with app.app_context():
            if status == "Approved":
                subject_line = f"✅ Khushal College - Leave Approved for {subject}"
                body = f"""
Dear {student_name},

Your leave application has been **APPROVED** at Khushal Degree College ✅

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

Your leave application has been **REJECTED** at Khushal Degree College ❌

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

# ==================================================
# 🔄 BACKGROUND EMAIL PROCESSING
# ==================================================

def send_attendance_emails_in_background(email_data_list):
    """
    Send attendance emails in background thread
    This prevents slow response times when saving manual attendance
    """
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
                        print(f"✅ Background Email {i+1}/{len(email_data_list)}: Sent to {email_data['student_email']}")
                    else:
                        fail_count += 1
                        print(f"❌ Background Email {i+1}/{len(email_data_list)}: Failed for {email_data['student_email']}")
                
                # Rate limiting: Wait 1 second between emails to avoid Gmail limits
                if i < len(email_data_list) - 1:  # Don't sleep after last email
                    time.sleep(1)
                    
            except Exception as e:
                fail_count += 1
                print(f"💥 Background Email {i+1}/{len(email_data_list)}: Error for {email_data['student_email']} - {e}")
        
        print(f"🎯 Background Email Task Completed: {success_count} successful, {fail_count} failed")
    
    # Start the email sending in a background thread
    thread = threading.Thread(target=email_worker)
    thread.daemon = True  # Thread will close when main app closes
    thread.start()

# ==================================================
# 🕒 AUTO-ABSENT SCHEDULER (STRICT & SMART)
# ==================================================

def mark_absentees_job():
    """
    STRICT LOGIC: 
    Runs every 1 minute.
    Checks for classes that ended within the last 5 minutes.
    Marks anyone NOT Present and NOT on Leave as 'Absent' IMMEDIATELY.
    """
    print("🕒 Scheduler: Checking for ended classes...")
    db = get_db_connection()
    if not db: return

    try:
        cursor = db.cursor(dictionary=True)
        now = datetime.now()
        date_today = now.date()
        current_time = now.strftime("%H:%M:%S")
        day_name = now.strftime("%A")
        
        # Calculate a 5-minute window to catch classes that just ended.
        # We look for classes ending between (Now - 5 mins) and (Now).
        # This prevents marking the same class twice or missing it by a few seconds.
        time_window_start = (now - timedelta(minutes=5)).strftime("%H:%M:%S")

        # 1️⃣ Find classes that JUST ended
        cursor.execute("""
            SELECT * FROM classes 
            WHERE day_of_week = %s 
            AND end_time <= %s 
            AND end_time > %s
        """, (day_name, current_time, time_window_start))
        
        ended_classes = cursor.fetchall()

        if not ended_classes:
            return # No classes ended in the last 5 mins

        for cls in ended_classes:
            class_id = cls['id']
            semester = cls['semester']
            subject = cls['subject_name']
            class_end = cls['end_time']
            
            print(f"🏁 Class Ended: {subject} ({semester}) at {class_end}. Marking absentees...")

            # 2️⃣ Find students who should be marked ABSENT
            # Logic: Belong to Semester AND (Not Present) AND (Not on Leave)
            cursor.execute("""
                SELECT id, name, email FROM students 
                WHERE semester = %s 
                AND status = 'approved'
                
                -- Exclude students who are already marked (Present OR Leave OR Absent)
                AND id NOT IN (
                    SELECT student_id FROM attendance 
                    WHERE date = %s AND class_id = %s
                )
                
                -- Exclude students who have an APPROVED LEAVE for today
                AND id NOT IN (
                    SELECT student_id FROM leaves 
                    WHERE status = 'Approved' 
                    AND %s BETWEEN start_date AND end_date
                )
            """, (semester, date_today, class_id, date_today))
            
            absentees = cursor.fetchall()
            
            # 3️⃣ Mark them Absent
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
            
            # Optional: Send emails if you have the email function
            if email_list:
                 # send_attendance_emails_in_background(email_list)
                 pass

    except Exception as e:
        print(f"💥 Scheduler Error: {e}")
    finally:
        if db: db.close()

# --- Start Scheduler (Runs every 1 minute) ---
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(mark_absentees_job, 'interval', minutes=1)
scheduler.start()

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
            body=f"""
🎉 CONGRATULATIONS!

Your Khushal Degree College Face Attendance System 
email notification system is now fully operational!

This is a secure and professional email system for your college!

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
# 🏠 MAIN ROUTES (UPDATED WITH UNIFIED AUTH)
# ==================================================

@app.route('/')
def index():
    if 'role' in session and session['role'] == 'admin':
        release_camera()
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'role' in session and session['role'] == 'admin':
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Read admin credentials from environment
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

        # 1. Total Approved Students
        cursor.execute("SELECT COUNT(*) AS total FROM students WHERE status='approved'")
        student_count = cursor.fetchone()
        total_students = student_count['total'] if student_count else 0

        # 2. Present Today
        today = datetime.now().date()
        cursor.execute("SELECT COUNT(DISTINCT student_id) AS present_today FROM attendance WHERE date = %s AND status = 'Present'", (today,))
        present_result = cursor.fetchone()
        present_today = present_result['present_today'] if present_result else 0

        # 3. Upcoming Class
        now_time = datetime.now().strftime("%H:%M:%S")
        current_day = datetime.now().strftime("%A")
        cursor.execute("SELECT subject_name FROM classes WHERE day_of_week = %s AND start_time > %s ORDER BY start_time ASC LIMIT 1", (current_day, now_time))
        upcoming_class = cursor.fetchone()
        upcoming_class_name = upcoming_class['subject_name'] if upcoming_class else "No More Classes"

        # 4. PENDING SIGNUPS (Students + Professors) - REPLACED LEAVES
        cursor.execute("SELECT COUNT(*) as count FROM students WHERE status='pending'")
        s_pending = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM professors WHERE status='pending'")
        p_pending = cursor.fetchone()['count']
        
        total_pending_signups = s_pending + p_pending

        return jsonify({
            "students": total_students,
            "present_today": present_today,
            "upcoming_class": upcoming_class_name,
            "pending_signups": total_pending_signups # Changed key from pending_leaves
        })
    except Exception as e:
        print("❌ Dashboard stats error:", e)
        return jsonify({"students": 0, "present_today": 0, "upcoming_class": "Error", "pending_signups": 0})
    finally:
        if cursor: cursor.close()
        if db: db.close()
# ==================================================
# 🎓 STUDENT MANAGEMENT (UPDATED WITH SEMESTER)
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
    return render_template('manage_students.html', students=students, selected_semester=sem_filter)

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
            semester = request.form.get('semester', '1st Semester')  # 👈 NEW: Semester field
            cursor.execute("UPDATE students SET name=%s, roll_no=%s, email=%s, semester=%s WHERE id=%s", 
                          (name, roll_no, email, semester, student_id))
            db.commit()
            return redirect(url_for('manage_students'))
        
        cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
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
    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT roll_no, name FROM students WHERE id=%s", (student_id,))
        student = cursor.fetchone()
        if student:
            cursor.execute("DELETE FROM students WHERE id=%s", (student_id,))
            db.commit()
            folder_name = f"{student['roll_no']}_{student['name'].replace(' ', '_')}"
            # Use relative path
            face_path = os.path.join(app.root_path, "faces", folder_name)
            if os.path.exists(face_path):
                import shutil
                shutil.rmtree(face_path)
                print(f"🗑 Deleted folder: {face_path}")
            
            # Trigger face cache reload
            load_known_faces()
            
    except Exception as e:
        if db:
            db.rollback()
        print(f"❌ Error deleting student: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()
            
    return redirect(url_for('manage_students'))

# ==================================================
# 🎓 STUDENT AUTHENTICATION (UPDATED WITH SEMESTER)
# ==================================================

@app.route('/student_signup', methods=['GET', 'POST'])
def student_signup():
    # ✅ FIX 1: Initialize db to None right at the start
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

            def decode_b64(b64):
                return base64.b64decode(b64.split(",")[1] if "," in b64 else b64)

            bytes_front = decode_b64(b64_front)
            bytes_left = decode_b64(b64_left)
            bytes_right = decode_b64(b64_right)

            # --- 🛡️ STRICT CHECKS ---
            
            # 1. Blur Check (Threshold 150)
            is_blur, score = is_image_blurry(bytes_front, threshold=150)
            if is_blur: return render_template('student_signup.html', error=f"Front photo is too blurry (Score: {int(score)}). Hold steady!")

            is_blur, score = is_image_blurry(bytes_left, threshold=150)
            if is_blur: return render_template('student_signup.html', error=f"Left photo is too blurry (Score: {int(score)}).")
            
            is_blur, score = is_image_blurry(bytes_right, threshold=150)
            if is_blur: return render_template('student_signup.html', error=f"Right photo is too blurry (Score: {int(score)}).")

            # 2. Face Size & Identity Check
            is_valid, msg = validate_three_angles(bytes_front, bytes_left, bytes_right)
            if not is_valid:
                return render_template('student_signup.html', error=msg)

            # --- DB & Save Logic ---
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            
            cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
            if cursor.fetchone():
                return render_template('student_signup.html', error="Roll Number already exists!")

            safe_name = name.replace(" ", "_")
            student_folder = os.path.join(app.root_path, "faces", f"{roll_no}_{safe_name}")
            os.makedirs(student_folder, exist_ok=True)

            with open(os.path.join(student_folder, "front.jpg"), "wb") as f: f.write(bytes_front)
            with open(os.path.join(student_folder, "left.jpg"), "wb") as f: f.write(bytes_left)
            with open(os.path.join(student_folder, "right.jpg"), "wb") as f: f.write(bytes_right)

            main_image_path = os.path.join(student_folder, "front.jpg")
            hashed_pw = generate_password_hash(password)
            
            cursor.execute("INSERT INTO students (name, roll_no, email, password, semester, image_path, status) VALUES (%s, %s, %s, %s, %s, %s, 'pending')", 
                           (name, roll_no, email, hashed_pw, semester, main_image_path))
            db.commit()
            
            load_known_faces()
            return render_template('student_signup.html', message="✅ Registration Successful!")

        except Exception as e:
            print(f"Error: {e}")
            return render_template('student_signup.html', error=f"Error: {e}")
        finally:
            # ✅ FIX 2: Now this works because db is defined as None at the top
            if db: 
                db.close()

    return render_template('student_signup.html')

    return render_template('student_signup.html')
@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if 'role' in session and session['role'] == 'student':
        return redirect(url_for('student_dashboard'))
        
    db = None
    cursor = None
    if request.method == 'POST':
        try:
            db = get_db_connection()
            if not db:
                return render_template('student_login.html', error="Database connection error")
            cursor = db.cursor(dictionary=True)

            roll_no = request.form['roll_no']
            password = request.form.get('password', '')
            cursor.execute("SELECT * FROM students WHERE roll_no = %s", (roll_no,))
            student = cursor.fetchone()
            
            if student:
                if student['password']:
                    # Check hashed password
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
                    # Legacy fallback for students without a password
                    session['logged_in'] = True
                    session['role'] = 'student'
                    session['user_id'] = student['id']
                    session['name'] = student['name']
                    return redirect(url_for('student_dashboard'))
            else:
                return render_template('student_login.html', error="Student not found")
                
        except Exception as e:
            print(f"❌ Error in student login: {e}")
            return render_template('student_login.html', error=f"An error occurred: {e}")
        finally:
            if cursor:
                cursor.close()
            if db:
                db.close()
                
    return render_template('student_login.html')

@app.route('/student_dashboard')
def student_dashboard():
    if 'role' not in session or session['role'] != 'student':
        return redirect(url_for('student_login'))
    
    student_id = session['user_id']
    attendance_data = []
    leave_requests = []
    
    db = None
    cursor = None
    
    try:
        db = get_db_connection()
        if not db:
            return "Database connection error", 500
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
        
    except Exception as e:
        print(f"❌ Error fetching student dashboard: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()
    
    return render_template('student_dashboard.html', 
                          attendance_data=attendance_data, 
                          leave_requests=leave_requests, 
                          student_name=session['name'])

@app.route('/student_logout')
def student_logout():
    session.clear()
    return redirect(url_for('student_login'))

# ==================================================
# ⚡ INSTANT PHOTO CHECK API   if blure or face not detected
# ==================================================
@app.route('/check_photo_quality', methods=['POST'])
def check_photo_quality():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({"valid": False, "error": "No image data received"})

        # Decode Base64
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
        
        image_bytes = base64.b64decode(encoded)

        # 1. Check Blur (Reuse your strict function)
        is_blur, score = is_image_blurry(image_bytes, threshold=150)
        if is_blur:
            return jsonify({"valid": False, "error": f"⚠️ Too Blurry (Score: {int(score)}). Please hold steady!"})

        # 2. Check Face Presence & Size
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        if not boxes:
            return jsonify({"valid": False, "error": "⚠️ No face detected. Look at the camera."})
        
        # Check Size (Prevent tiny faces/cars)
        top, right, bottom, left = boxes[0]
        face_area = (bottom - top) * (right - left)
        h, w, _ = img.shape
        image_area = h * w
        
        if face_area < (image_area * 0.08):
             return jsonify({"valid": False, "error": "⚠️ Face too small. Please move closer."})

        return jsonify({"valid": True})

    except Exception as e:
        print(f"Validation Error: {e}")
        return jsonify({"valid": False, "error": "Server error processing image"})

# ==================================================
# 👨‍🏫 PROFESSOR MANAGEMENT (UPDATED)
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
            # Check if email exists
            cursor.execute("SELECT id FROM professors WHERE email=%s", (email,))
            if cursor.fetchone():
                return render_template('professor_signup.html', error="❌ Email already registered!")
            
            # Insert with 'pending' status
            cursor.execute("INSERT INTO professors (name, email, password, status) VALUES (%s, %s, %s, 'pending')", 
                           (name, email, hashed_pw))
            db.commit()
            return render_template('professor_signup.html', message="✅ Application Sent! Please wait for Admin approval.")
        except Exception as e:
            return render_template('professor_signup.html', error=f"Error: {e}")
        finally:
            if cursor: cursor.close()
            if db: db.close()
            
    return render_template('professor_signup.html')

@app.route('/professor_login', methods=['GET', 'POST'])
def professor_login():
    if 'role' in session and session['role'] == 'professor':
        return redirect(url_for('professor_dashboard'))
        
    db = None
    cursor = None
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
            # Check hash or plain text (for legacy support)
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
    """List approved professors."""
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    # Only show approved professors. Pending ones are in 'view_leaves'.
    cursor.execute("SELECT * FROM professors WHERE status='approved' ORDER BY name")
    professors = cursor.fetchall()
    db.close()
    return render_template('manage_professors.html', professors=professors)

@app.route('/edit_professor/<int:id>', methods=['GET', 'POST'])
@admin_required
def edit_professor(id):
    """Edit professor details."""
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
    
    if not professor:
        return "Professor not found", 404
        
    return render_template('edit_professor.html', professor=professor)

@app.route('/delete_professor/<int:id>')
@admin_required
def delete_professor(id):
    """Delete a professor."""
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("DELETE FROM professors WHERE id=%s", (id,))
    db.commit()
    db.close()
    return redirect(url_for('manage_professors'))
# ==================================================
# ✏️ EDIT CLASS ROUTE
# ==================================================
@app.route('/edit_class/<int:class_id>', methods=['GET', 'POST'])
@admin_required
def edit_class(class_id):
    db = get_db_connection()
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
            return redirect(url_for('manage_classes'))
            
        except Exception as e:
            print(f"Error updating class: {e}")
            if db: db.rollback()
    
    # GET: Fetch class details & professors for dropdown
    cursor.execute("SELECT * FROM classes WHERE id=%s", (class_id,))
    class_info = cursor.fetchone()
    
    cursor.execute("SELECT id, name FROM professors WHERE status='approved'")
    professors = cursor.fetchall()
    
    db.close()
    
    if not class_info:
        return "Class not found", 404
        
    # Convert timedelta/time objects to string for HTML input (HH:MM)
    def format_time(t):
        if hasattr(t, 'seconds'): # If it's a timedelta
            seconds = t.seconds
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"{h:02}:{m:02}"
        return str(t)

    class_info['start_time'] = format_time(class_info['start_time'])
    class_info['end_time'] = format_time(class_info['end_time'])

    return render_template('edit_class.html', class_info=class_info, professors=professors)

# ==================================================
# 🏫 CLASS MANAGEMENT (UPDATED WITH SEMESTER)
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
            semester = request.form.get('semester', '1st Semester')  # 👈 NEW: Semester field
            day_of_week = request.form['day_of_week']
            start_time = request.form['start_time']
            end_time = request.form['end_time']
            cursor.execute("INSERT INTO classes (subject_name, professor_id, semester, day_of_week, start_time, end_time) VALUES (%s, %s, %s, %s, %s, %s)", 
                          (subject_name, professor_id, semester, day_of_week, start_time, end_time))
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
    semester = request.args.get('semester', 'all') # 👈 Get semester from dropdown
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    # Base Query: Join Students, Attendance, and Classes
    # We use LEFT JOIN so we can see students even if they have no attendance (optional)
    # But for reports, usually JOIN is better to see actual records.
    
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
    
    # 👇 FIX: Apply the Semester Filter
    if semester != 'all':
        base_query += " AND s.semester = %s"
        params.append(semester)
        
    # Apply Subject Filter
    if subject != 'all':
        base_query += " AND c.subject_name = %s"
        params.append(subject)
        
    # Apply Date Period Filter (Smart Logic)
    if period == 'day':
        base_query += " AND a.date = CURDATE()"
    elif period == 'week':
        base_query += " AND YEARWEEK(a.date, 1) = YEARWEEK(CURDATE(), 1)"
    elif period == 'month':
        base_query += " AND MONTH(a.date) = MONTH(CURDATE()) AND YEAR(a.date) = YEAR(CURDATE())"

    base_query += " GROUP BY s.id, c.subject_name ORDER BY s.semester, s.roll_no"
    
    cursor.execute(base_query, tuple(params))
    data = cursor.fetchall()
    
    # Calculate Percentage
    for row in data:
        if row['total_classes'] > 0:
            row['percentage'] = round((row['presents'] / row['total_classes']) * 100, 1)
        else:
            row['percentage'] = 0
            
    db.close()
    return jsonify(data)

# ==================================================
# 📅 WEEKLY ATTENDANCE API (ADMIN PAGE)
# ==================================================

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
    
    # 1. Fetch Students
    cursor.execute("SELECT id, name, roll_no FROM students WHERE semester = %s ORDER BY roll_no", (semester,))
    students = cursor.fetchall()
    
    # 2. Fetch Weekly Logs
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
    
    # 3. Fetch OVERALL Statistics (For the % column)
    # This query calculates the total percentage for this semester/subject context
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

    # 4. Map Data
    attendance_map = {}
    for log in logs:
        sid = log['student_id']
        date_key = str(log['date'])
        if sid not in attendance_map: attendance_map[sid] = {}
        attendance_map[sid][date_key] = {'status': log['status'], 'time': str(log['time'])}

    # 5. Build Final Response
    final_data = []
    for s in students:
        # Calculate %
        stat = stats_data.get(s['id'], {'p': 0, 't': 0})
        pct = round((stat['p'] / stat['t']) * 100) if stat['t'] > 0 else 0
        
        final_data.append({
            'name': s['name'],
            'roll': s['roll_no'],
            'week_data': attendance_map.get(s['id'], {}),
            'overall_percent': pct
        })
        
    db.close()
    return jsonify(final_data)

# ==================================================
# 📅 PROFESSOR WEEKLY API (PROFESSOR PAGE)
# ==================================================

@app.route('/get_professor_weekly_attendance')
@professor_required
def get_professor_weekly_attendance():
    semester = request.args.get('semester')
    subject = request.args.get('subject')
    start_date_str = request.args.get('start_date')
    professor_id = session['user_id']
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    # 1. Date Range
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = start_date + timedelta(days=6)
    
    # 2. Fetch Students in this Semester
    cursor.execute("SELECT id, name, roll_no FROM students WHERE semester = %s ORDER BY roll_no", (semester,))
    students = cursor.fetchall()
    
    # 3. Fetch Attendance Logs (Only for this Professor's classes)
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
    
    # 4. Fetch Overall Stats (For the % Badge)
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

    # 5. Map Data
    attendance_map = {}
    for log in logs:
        sid = log['student_id']
        date_key = str(log['date'])
        if sid not in attendance_map: attendance_map[sid] = {}
        # Priority: If multiple classes per day, usually show the latest or specific subject
        attendance_map[sid][date_key] = {'status': log['status'], 'time': str(log['time'])}

    # 6. Build Response
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
        
    db.close()
    return jsonify(final_data)
# ==================================================
# 🎥 LIVE ATTENDANCE (UPDATED WITH SEMESTER FILTER)
# ==================================================

@app.route('/live_attendance')
@admin_required
def live_attendance():
    return render_template('live_attendance.html')

def generate_frames():
    global camera, camera_active, KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_ROLLS
    
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
        camera_active = True

    print("🎥 Camera started for live attendance")
    
    db = None
    cursor = None
    
    try:
        db = get_db_connection()
        if not db:
            print("❌ Live Attendance: Cannot connect to DB.")
            return
        cursor = db.cursor(dictionary=True)

        while camera_active:
            success, frame = camera.read()
            if not success:
                print("❌ Failed to read frame from camera")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, faces)

            now = datetime.now()
            date = now.date()
            time_now = now.strftime("%H:%M:%S")
            current_day = now.strftime("%A")

            # Check for DB connection liveness
            if not db.is_connected():
                print("ℹ️ Reconnecting to DB in live feed...")
                if cursor: cursor.close()
                if db: db.close()
                db = get_db_connection()
                cursor = db.cursor(dictionary=True)

            cursor.execute("SELECT * FROM classes WHERE day_of_week = %s AND start_time <= %s AND end_time >= %s LIMIT 1", (current_day, time_now, time_now))
            current_class = cursor.fetchone()

            subject = current_class['subject_name'] if current_class else "No Class"
            class_id = current_class['id'] if current_class else None
            semester = current_class.get('semester', '1st Semester') if current_class else None  # 👈 NEW: Get semester
            class_color = (255, 255, 0) if current_class else (0, 0, 255)
            cv2.putText(frame, f"Class: {subject} ({semester})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, class_color, 3)

            for encode, loc in zip(encs, faces):
                # Use the cached face encodings
                matches = face_recognition.compare_faces(KNOWN_ENCODINGS, encode)
                face_distances = face_recognition.face_distance(KNOWN_ENCODINGS, encode)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    y1, x2, y2, x1 = loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    if matches[best_match_index] and best_distance < 0.6:
                        name = KNOWN_NAMES[best_match_index]
                        roll = KNOWN_ROLLS[best_match_index]
                        color, label = (0, 255, 0), f"{name} ({roll})"

                        if current_class:
                            cursor.execute("SELECT id FROM students WHERE roll_no=%s", (roll,))
                            student = cursor.fetchone()
                            if student:
                                student_id = student['id']
                                cursor.execute("SELECT * FROM attendance WHERE student_id=%s AND date=%s AND class_id=%s", (student_id, date, class_id))
                                existing_attendance = cursor.fetchone()
                                
                                if not existing_attendance:
                                    cursor.execute("INSERT INTO attendance (student_id, date, time, status, class_id) VALUES (%s, %s, %s, %s, %s)", (student_id, date, time_now, "Present", class_id))
                                    db.commit()
                                    
                                    cursor.execute("SELECT email FROM students WHERE id = %s", (student_id,))
                                    student_data = cursor.fetchone()
                                    if student_data and student_data['email']:
                                        send_attendance_notification(student_data['email'], name, "Present", subject, date, time_now)
                                    
                                    message = f"✅ {name} ({roll}) marked Present for {subject}"
                                    update_detection(name, roll, subject, "present", message)
                                    print(message)
                                else:
                                    message = f"ℹ️ {name} ({roll}) already attended {subject}"
                                    update_detection(name, roll, subject, "already_attended", message)
                    else:
                        color, label = (0, 0, 255), "Unknown Face"
                        message = "⚠️ Unknown Face Detected!"
                        update_detection("Unknown", "Unknown", subject, "unknown", message)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    except Exception as e:
        print(f"❌ Error in generate_frames: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()
        release_camera()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    release_camera()
    return jsonify({'status': 'camera_stopped'})

# ==================================================
# 📝 LEAVE MANAGEMENT (UPDATED)
# ==================================================

@app.route('/apply_leave', methods=['GET', 'POST'])
def apply_leave():
    if 'role' not in session or session['role'] != 'student':
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

            cursor.execute("INSERT INTO leaves (student_id, subject_name, application_purpose, application_text, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s, %s, 'Pending')", (logged_in_student_id, subject_name, application_purpose, application_text, start_date, end_date))
            db.commit()

            # Send confirmation email to student
            if student and student['email']:
                try:
                    with app.app_context():
                        msg = Message(
                            subject="📝 Leave Application Submitted Successfully",
                            recipients=[student['email']],
                            body=f"""
Dear {student['name']},

Your leave application has been submitted successfully and is pending approval.

📚 Subject: {subject_name or 'All Subjects'}
🎯 Purpose: {application_purpose}
📅 Period: {start_date} to {end_date}

Best regards,
Face Attendance System
Khushal Degree College
                            """
                        )
                        mail.send(msg)
                        print(f"📧 Leave application confirmation sent to {student['email']}")
                except Exception as e:
                    print(f"❌ Failed to send leave confirmation email: {e}")

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
                load_known_faces() # Reload AI models
                
        elif req_type == 'professor':
            pid = request.form.get('professor_id')
            status = 'approved' if action == 'approve' else 'rejected'
            if action == 'reject':
                cursor.execute("DELETE FROM professors WHERE id=%s", (pid,))
            else:
                cursor.execute("UPDATE professors SET status=%s WHERE id=%s", (status, pid))

        db.commit()
        return redirect(url_for('view_requests'))

    # Fetch Pending Signups Only
    cursor.execute("SELECT * FROM students WHERE status='pending'")
    pending_students = cursor.fetchall()
    
    cursor.execute("SELECT * FROM professors WHERE status='pending'")
    pending_professors = cursor.fetchall()
    
    db.close()
    return render_template('view_requests.html', 
                           pending_students=pending_students,
                           pending_professors=pending_professors)

# ==========================================
# 📨 PROFESSOR: Manage Leaves (New Route)
# ==========================================
@app.route('/professor_leaves', methods=['GET', 'POST'])
@professor_required
def professor_leaves():
    professor_id = session['user_id']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    # 1. Handle Approval/Rejection
    if request.method == 'POST':
        leave_id = request.form.get('leave_id')
        action = request.form.get('action') 
        
        cursor.execute("UPDATE leaves SET status=%s WHERE id=%s", (action, leave_id))
        
        # (Optional) Add your auto-attendance marking logic here if desired
            
        db.commit()
        flash(f"Leave {action} successfully!", "success")
        return redirect(url_for('professor_leaves'))

    # 2. SMART FETCH: Handle "All Subjects" & Specific Subjects
    # We join distinct class data to ensure we don't get duplicate rows
    cursor.execute("""
        SELECT DISTINCT l.*, s.name, s.roll_no, s.semester 
        FROM leaves l
        JOIN students s ON l.student_id = s.id
        WHERE l.status = 'Pending'
        AND (
            -- Case A: The leave is for a specific subject this professor teaches
            l.subject_name IN (
                SELECT subject_name FROM classes WHERE professor_id = %s
            )
            OR
            -- Case B: The leave is for "All Subjects" (Empty) AND the professor teaches this Semester
            (
                (l.subject_name IS NULL OR l.subject_name = '')
                AND s.semester IN (
                    SELECT semester FROM classes WHERE professor_id = %s
                )
            )
        )
        ORDER BY l.start_date DESC
    """, (professor_id, professor_id))
    
    leave_records = cursor.fetchall()
    db.close()

    return render_template('professor_leaves.html', leave_records=leave_records)
# ==================================================
# 📋 MANUAL ATTENDANCE (UPDATED WITH SEMESTER FILTER)
# ==================================================

@app.route('/manual_attendance')
def manual_attendance():
    """Shows manual attendance page with semester filtering"""
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

@app.route('/get_class_students/<int:class_id>')
def get_class_students(class_id):
    """
    INTELLIGENT API:
    Fetches ONLY students belonging to the Class's Semester.
    """
    db = None
    cursor = None
    students = []
    existing_attendance = {}
    
    try:
        db = get_db_connection()
        if not db:
            return jsonify({"error": "Database connection error"}), 500
        cursor = db.cursor(dictionary=True)
        
        # 1. Get Class Semester
        cursor.execute("SELECT semester FROM classes WHERE id=%s", (class_id,))
        cls = cursor.fetchone()
        if not cls:
            return jsonify({'students': []})
        
        semester = cls['semester']
        
        # 2. Get Students in that Semester
        cursor.execute("SELECT id, name, roll_no FROM students WHERE semester=%s AND status='approved' ORDER BY name", (semester,))
        students = cursor.fetchall()
        
        # 3. Get existing attendance for today
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
    """
    Optimized manual attendance with background email processing
    """
    data = request.json
    class_id = data['class_id']
    date = data['date']
    attendance_data = data['attendance']
    
    print(f"📊 Manual Attendance: Saving {len(attendance_data)} students...")
    
    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return jsonify({'success': False, 'error': 'Database connection error'})
        cursor = db.cursor(dictionary=True)

        # Get class information once
        cursor.execute("SELECT start_time, subject_name FROM classes WHERE id = %s", (class_id,))
        class_info = cursor.fetchone()
        class_time = class_info['start_time']
        subject_name = class_info['subject_name']
        
        # STEP 1: Quickly save all attendance records first
        for student_id, status in attendance_data.items():
            cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, class_id, date))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("UPDATE attendance SET status = %s, time = %s, method = 'manual' WHERE id = %s", (status, class_time, existing['id']))
            else:
                cursor.execute("INSERT INTO attendance (student_id, class_id, date, time, status, method) VALUES (%s, %s, %s, %s, %s, 'manual')", (student_id, class_id, date, class_time, status))
        
        # Commit attendance data immediately
        db.commit()
        print(f"✅ Attendance saved successfully for {len(attendance_data)} students")
        
        # STEP 2: Prepare email data for background processing
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
        
        # STEP 3: Start background email processing
        if email_data_list:
            print(f"📧 Starting background email process for {len(email_data_list)} students...")
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
    """
    Optimized bulk attendance with background email processing
    """
    data = request.json
    action = data['action']
    student_ids = data['student_ids']
    class_id = data['class_id']
    date = data['date']
    
    print(f"📊 Bulk Attendance: {action} for {len(student_ids)} students...")
    
    db = None
    cursor = None
    try:
        db = get_db_connection()
        if not db:
            return jsonify({'success': False, 'error': 'Database connection error'})
        cursor = db.cursor(dictionary=True)

        # Get class information
        cursor.execute("SELECT subject_name FROM classes WHERE id = %s", (class_id,))
        subject_name = cursor.fetchone()['subject_name']
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # STEP 1: Quickly save all attendance records
        for student_id in student_ids:
            cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND class_id = %s AND date = %s", (student_id, class_id, date))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("UPDATE attendance SET status = %s WHERE id = %s", (action, existing['id']))
            else:
                cursor.execute("INSERT INTO attendance (student_id, class_id, date, time, status, method) VALUES (%s, %s, %s, %s, %s, 'manual')", (student_id, class_id, date, current_time, action))
        
        db.commit()
        print(f"✅ Bulk attendance saved successfully")
        
        # STEP 2: Prepare email data for background processing
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
            
            # STEP 3: Start background email processing
            if email_data_list:
                print(f"📧 Starting background email process for {len(email_data_list)} students...")
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
# 📁 UTILITY ROUTES
# ==================================================

@app.route('/face_images/<path:filename>')
def face_images(filename):
    # Use relative path
    face_dir = os.path.join(app.root_path, "faces")
    return send_from_directory(face_dir, filename)

# ==================================================
# 👨‍🏫 PROFESSOR AUTHENTICATION & MANAGEMENT (UPDATED)
# ==================================================

@app.route('/professor_set_password', methods=['GET', 'POST'])
def professor_set_password():
    if request.method == 'GET':
        professor_id = request.args.get('professor_id')
        email = request.args.get('email')
        if not professor_id or not email:
            return redirect(url_for('professor_login'))
        return render_template('professor_set_password.html', 
                             professor_id=professor_id, 
                             email=email)
    
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
            return render_template('professor_set_password.html', 
                                 error="Passwords do not match!",
                                 professor_id=professor_id, email=email)
        
        if len(password) < 6:
            return render_template('professor_set_password.html',
                                 error="Password must be at least 6 characters!",
                                 professor_id=professor_id, email=email)
        
        # Hash the password
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

@app.route('/professor_dashboard')
@professor_required
def professor_dashboard():
    professor_id = session['user_id']
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    # 1. Fetch Professor Details (Name & Email)
    cursor.execute("SELECT name, email FROM professors WHERE id = %s", (professor_id,))
    prof_data = cursor.fetchone()

    # 2. Get Today's Classes (for the "Your Classes" and "Next Class" cards)
    today_name = datetime.now().strftime("%A")
    cursor.execute("""
        SELECT * FROM classes 
        WHERE professor_id = %s AND day_of_week = %s 
        ORDER BY start_time ASC
    """, (professor_id, today_name))
    todays_classes = cursor.fetchall()

    # 3. Get "Present Today" Count
    # Counts how many students are marked 'Present' today in THIS professor's classes
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM attendance a
        JOIN classes c ON a.class_id = c.id
        WHERE c.professor_id = %s 
        AND a.date = CURDATE() 
        AND a.status = 'Present'
    """, (professor_id,))
    present_count = cursor.fetchone()['count']

    # 4. Get "Pending Leaves" Count
    # Counts total pending leaves (you can refine this to specific semesters if needed)
    cursor.execute("SELECT COUNT(*) as count FROM leaves WHERE status = 'Pending'")
    leaves_count = cursor.fetchone()['count']

    # 5. Determine "Next Class" Logic
    current_time = datetime.now().time()
    next_class = None
    for cls in todays_classes:
        # Convert string time to time object if necessary
        # Assuming db returns timedelta or time, logic handles basic comparison
        # (This depends on your DB driver, simplified here for display)
        start_t = (datetime.min + cls['start_time']).time() if isinstance(cls['start_time'], timedelta) else cls['start_time']
        if start_t > current_time:
            next_class = cls
            break

    db.close()

    return render_template('professor_dashboard.html', 
                           professor=prof_data,
                           classes=todays_classes,
                           present_count=present_count,
                           leaves_count=leaves_count,
                           next_class=next_class)
@app.route('/professor_logout')
def professor_logout():
    session.clear()
    return redirect(url_for('professor_login'))

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
    # 1. Get Filters
    subject = request.args.get('subject', 'all')
    semester = request.args.get('semester', 'all')
    professor_id = session['user_id']
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    # 2. Base Query: Only show students for classes THIS professor teaches
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
    
    # 3. Apply Filters
    if subject != 'all':
        query += " AND c.subject_name = %s"
        params.append(subject)
        
    if semester != 'all':
        query += " AND s.semester = %s"
        params.append(semester)
        
    query += " GROUP BY s.id, c.subject_name ORDER BY s.semester, s.roll_no"
    
    cursor.execute(query, tuple(params))
    data = cursor.fetchall()
    
    # 4. Calculate Percentage
    for row in data:
        if row['total_classes'] > 0:
            row['percentage'] = round((row['presents'] / row['total_classes']) * 100, 1)
        else:
            row['percentage'] = 0
            
    db.close()
    return jsonify(data)

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
    return render_template('professor_manual_attendance.html', 
                         classes=formatted_classes, 
                         today=today)



@app.route('/professor_approve_leave', methods=['POST'])
@professor_required
def professor_approve_leave():
    leave_id = request.form['leave_id']
    action = request.form['action'] # 'Approved' or 'Rejected'
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    try:
        # 1. Fetch Leave & Student Details
        cursor.execute("""
            SELECT l.*, s.name, s.email, s.semester 
            FROM leaves l 
            JOIN students s ON l.student_id = s.id 
            WHERE l.id = %s
        """, (leave_id,))
        leave = cursor.fetchone()
        
        if not leave:
            return jsonify({'success': False, 'error': 'Leave not found'})
        
        # 2. Update Leave Status
        cursor.execute("UPDATE leaves SET status = %s WHERE id = %s", (action, leave_id))
        
        # 3. IF APPROVED: Automatically Insert "Leave" into Attendance Table
        if action == 'Approved':
            student_id = leave['student_id']
            subject_name = leave['subject_name'] # Could be specific subject or None
            semester = leave['semester']
            start_date = leave['start_date'] # Check if these are date objects or strings
            end_date = leave['end_date']

            # Ensure dates are Python Date objects
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

            # Loop through every day of the leave
            current_date = start_date
            while current_date <= end_date:
                # Find which class IDs to mark (Specific Subject OR All Classes for that Semester)
                if subject_name:
                    # Find specific class for this subject and semester
                    cursor.execute("""
                        SELECT id FROM classes 
                        WHERE subject_name = %s AND semester = %s
                    """, (subject_name, semester))
                else:
                    # If "All Subjects", find ALL classes for this semester
                    cursor.execute("""
                        SELECT id FROM classes 
                        WHERE semester = %s
                    """, (semester,))
                
                target_classes = cursor.fetchall()

                # Insert 'Leave' for each relevant class
                for cls in target_classes:
                    class_id = cls['id']
                    
                    # Delete existing record first (to overwrite Absent/Present)
                    cursor.execute("""
                        DELETE FROM attendance 
                        WHERE student_id = %s AND class_id = %s AND date = %s
                    """, (student_id, class_id, current_date))
                    
                    # Insert LEAVE record
                    cursor.execute("""
                        INSERT INTO attendance (student_id, class_id, date, time, status, method)
                        VALUES (%s, %s, %s, NOW(), 'Leave', 'system')
                    """, (student_id, class_id, current_date))
                    
                    print(f"✅ Marked Leave: Student {student_id}, Class {class_id}, Date {current_date}")

                # Move to next day
                current_date += timedelta(days=1)

        db.commit()
        
        # 4. Send Email Notification (Optional)
        if leave['email']:
            try:
                # Assuming you have this function defined elsewhere or imported
                # send_leave_status_notification(...) 
                pass 
            except:
                pass

        return jsonify({'success': True, 'message': f'Leave {action} and attendance updated!'})

    except Exception as e:
        if db: db.rollback()
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if cursor: cursor.close()
        if db: db.close()

# ==================================================
# 🏁 CLEANUP & STARTUP
# ==================================================

@atexit.register
def cleanup_on_exit():
    release_camera()
    if scheduler.running:
        scheduler.shutdown()
    print("Application exiting. Camera released and scheduler stopped.")

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    )