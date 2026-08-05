import cv2
import os
import mysql.connector

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_attendance_db"
)
cursor = db.cursor()

# Ensure the 'faces' folder exists
if not os.path.exists("faces"):
    os.makedirs("faces")

# Get student info
name = input("Enter Student Name: ")
roll_no = input("Enter Roll Number: ")

# Create a folder for this student's photos
student_folder = f"faces/{roll_no}_{name.replace(' ', '_')}"
os.makedirs(student_folder, exist_ok=True)

# Start webcam
cam = cv2.VideoCapture(0)
cv2.namedWindow("Face Capture")

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Face Capture", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC key to exit
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:  # SPACE key to capture
        img_name = f"{student_folder}/image_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"✅ {img_name} saved!")
        count += 1

        if count >= 5:
            print("Captured 5 images — done!")
            break

cam.release()
cv2.destroyAllWindows()

# Save student info in database
image_path = student_folder
sql = "INSERT INTO students (name, roll_no, image_path) VALUES (%s, %s, %s)"
val = (name, roll_no, image_path)
cursor.execute(sql, val)
db.commit()

print("✅ Student registered successfully!")

cursor.close()
db.close()
