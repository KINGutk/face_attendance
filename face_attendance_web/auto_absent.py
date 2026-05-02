import mysql.connector
from datetime import datetime, timedelta
import time

# --- DATABASE CONNECTION ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_attendance_db"
)
cursor = db.cursor(dictionary=True)

print("🕒 Auto Absent Scheduler started... (checks every 5 minutes)")

def mark_absentees():
    now = datetime.now()
    date_today = now.date()
    current_time = now.strftime("%H:%M:%S")
    current_day = now.strftime("%A")

    # 1️⃣ Get all classes that ended at least 15 minutes ago but not processed yet
    cursor.execute("""
        SELECT * FROM classes
        WHERE day_of_week = %s
        AND ADDTIME(end_time, '00:15:00') <= %s
    """, (current_day, current_time))
    classes = cursor.fetchall()

    if not classes:
        print("📭 No classes eligible for absent marking right now.\n")
        return

    for cls in classes:
        class_id = cls['id']
        subject = cls['subject_name']
        class_end = cls['end_time']

        print(f"\n📘 Checking absentees for {subject} (ended at {class_end})...")

        # 2️⃣ Find all students who were not marked present in this class
        cursor.execute("""
            SELECT s.id AS student_id, s.name
            FROM students s
            WHERE s.id NOT IN (
                SELECT a.student_id FROM attendance a
                WHERE a.date = %s AND a.class_id = %s
            )
        """, (date_today, class_id))
        students = cursor.fetchall()

        if not students:
            print("✅ All students have attendance marked.\n")
            continue

        for student in students:
            student_id = student['student_id']
            student_name = student['name']

            # 3️⃣ Skip if student is on approved leave during this date
            cursor.execute("""
                SELECT * FROM leaves
                WHERE student_id=%s AND status='Approved'
                AND %s BETWEEN start_date AND end_date
            """, (student_id, date_today))
            leave = cursor.fetchone()

            if leave:
                print(f"🟨 Skipping {student_name} — on approved leave.")
                continue

            # 4️⃣ Mark Absent
            cursor.execute("""
                INSERT INTO attendance (student_id, date, time, status, class_id)
                VALUES (%s, %s, %s, %s, %s)
            """, (student_id, date_today, class_end, "Absent", class_id))
            db.commit()
            print(f"❌ Marked Absent: {student_name}")

        print(f"✅ Absentees marked for {subject} class.\n")


# --- MAIN LOOP ---
try:
    while True:
        mark_absentees()
        time.sleep(300)  # 5 minutes delay
except KeyboardInterrupt:
    print("🛑 Auto Absent Scheduler stopped manually.")
finally:
    cursor.close()
    db.close()
