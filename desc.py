import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv('D:/face_attendance/.env')

try:
    db = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASS', ''),
        database=os.getenv('DB_NAME', 'face_attendance')
    )
    cursor = db.cursor(dictionary=True)
    cursor.execute("DESCRIBE leaves")
    for row in cursor.fetchall():
        print(f"{row['Field']} - {row['Type']}")
except Exception as e:
    print(f"Error: {e}")
