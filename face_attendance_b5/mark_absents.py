# ---------------- AUTO ABSENT MARKER ----------------
import mysql.connector
from datetime import datetime

# Connect to database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_attendance_db"
)
cursor = db.cursor(dictionary=True)

now = datetime.now()
today = now.date()
current_time = now.time()

# 1️⃣ Find which class has ended recently (within last 5 minutes)
cursor.execute("""
    SELECT * FROM classes
    WHERE end_time <= %s
    ORDER BY end_time DESC
    LIMIT 1
""", (current_time,))
recent_class = cursor.fetchone()

if recent_class:
    class_id = recent_class['id']
    class_name = recent_class['subject_name']

    print(f"📘 Checking absentees for {class_name} class...")

    # 2️⃣ Get all students
    cursor.execute("SELECT id, name FROM students")
    all_students = cursor.fetchall()

    # 3️⃣ Get those marked "Present"
    cursor.execute("""
        SELECT DISTINCT student_id FROM attendance
        WHERE date=%s AND class_id=%s
    """, (today, class_id))
    present_students = {row['student_id'] for row in cursor.fetchall()}

    # 4️⃣ Mark "Absent" for others
    absents = [s for s in all_students if s['id'] not in present_students]
    for s in absents:
        cursor.execute("""
            INSERT INTO attendance (student_id, class_id, date, time, status)
            VALUES (%s, %s, %s, %s, %s)
        """, (s['id'], class_id, today, recent_class['end_time'], "Absent"))
        print(f"❌ Marked absent: {s['name']}")

    db.commit()
    print(f"✅ Absentees marked for {class_name} class.")
else:
    print("⚠️ No class recently ended.")

cursor.close()
db.close()
