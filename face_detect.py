import cv2
import os
import threading
import datetime
import pyttsx3
from tkinter import *
from tkinter import filedialog

# Load Haar classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Globals
cap = None
running = False
latest_faces = []
save_folder = ""

# Function to save detected faces
def save_faces(frame, faces):
    global save_folder
    if not save_folder:
        save_folder = filedialog.askdirectory(title="Choose folder to save faces")
        if not save_folder:
            return
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, (x, y, w, h) in enumerate(faces):
        roi = frame[y:y+h, x:x+w]
        filename = os.path.join(save_folder, f"face_{now}_{i}.jpg")
        cv2.imwrite(filename, roi)
        print(f"[✔] Saved: {filename}")

# Face detection thread
def detect_faces():
    global cap, running, latest_faces
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Faster on Windows

    if not cap.isOpened():
        print("Camera not available")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        latest_faces = faces
        face_label.config(text=f"Faces Detected: {len(faces)}")

        if len(faces) > 0:
            engine.say(f"{len(faces)} face{'s' if len(faces) != 1 else ''} detected")
            engine.runAndWait()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)

        cv2.imshow("Face Detection by soumo chandra", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Button actions
def start_detection():
    global running
    if not running:
        running = True
        threading.Thread(target=detect_faces).start()

def stop_detection():
    global running
    running = False

def save_detected_faces():
    if cap is not None and len(latest_faces) > 0:
        ret, frame = cap.read()
        if ret:
            save_faces(frame, latest_faces)

def close_app():
    stop_detection()
    root.destroy()

# GUI setup
root = Tk()
root.title("Face Detection by soumo chandra")
root.geometry("360x300")
root.configure(bg="#f0f0f0")

Label(root, text="Face Detection App by soumo chandra", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333").pack(pady=15)
face_label = Label(root, text="Faces Detected: 0", font=("Arial", 12), bg="#f0f0f0", fg="#444")
face_label.pack(pady=5)

btn_style = {"font": ("Arial", 12), "bg": "#007acc", "fg": "white", "activebackground": "#005f99", "width": 20}

Button(root, text="Start Detection", command=start_detection, **btn_style).pack(pady=5)
Button(root, text="Stop Detection", command=stop_detection, **btn_style).pack(pady=5)
Button(root, text="Save Faces", command=save_detected_faces, **btn_style).pack(pady=5)
Button(root, text="Exit", command=close_app, **btn_style).pack(pady=10)

root.mainloop()
