file_path = "D:/face_attendance/face_attendance_web/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """        WHERE l.status != 'Pending'
        AND (
            l.subject_name IN (
                SELECT subject_name FROM classes WHERE professor_id = %s
            )
            OR l.semester IN (
                SELECT semester FROM classes WHERE professor_id = %s
            )
        )
        ORDER BY l.start_date DESC"""

replacement = """        WHERE l.status != 'Pending'
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
        ORDER BY l.start_date DESC"""

if target in content:
    content = content.replace(target, replacement)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Success")
else:
    print("Target not found")
