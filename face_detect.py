import cv2
import os
import threading
import datetime
import pyttsx3
from tkinter import *
from tkinter import filedialog

# Haar Classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Globals
cap = None
running = False
latest_faces = []
save_folder = ""
last_spoken_emotion = ""

# Save faces function
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

# Detection thread
def detect_faces():
    global cap, running, latest_faces, last_spoken_emotion

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera not found!")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        latest_faces = faces
        face_label.config(text=f"Faces Detected: {len(faces)}")

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Eyes detection
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)

            # Smile detection (emotion)
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)

            if len(smiles) > 0:
                emotion = "Smiling 😄"
                if last_spoken_emotion != "smile":
                    last_spoken_emotion = "smile"
                    engine.say("You are smiling")
                    engine.runAndWait()
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
            else:
                emotion = "Neutral 😐"
                if last_spoken_emotion != "neutral":
                    last_spoken_emotion = "neutral"

            # GUI emotion update
            emotion_label.config(text=f"Emotion: {emotion}")
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Face & Smile Detection by Soumo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Buttons functionality
def start_detection():
    global running
    if not running:
        running = True
        threading.Thread(target=detect_faces, daemon=True).start()

def stop_detection():
    global running
    running = False

def save_detected_faces():
    if cap and len(latest_faces) > 0:
        ret, frame = cap.read()
        if ret:
            save_faces(frame, latest_faces)

def close_app():
    stop_detection()
    root.destroy()

# GUI
root = Tk()
root.title("Face Emotion App by Soumo")
root.geometry("400x350")
root.configure(bg="#e6f2ff")

Label(root, text="Smile Emotion Detection App 😊",
      font=("Arial", 16, "bold"),
      bg="#e6f2ff", fg="#003366").pack(pady=15)

face_label = Label(root, text="Faces Detected: 0",
                   font=("Arial", 12),
                   bg="#e6f2ff", fg="#333")
face_label.pack(pady=5)

emotion_label = Label(root, text="Emotion: None",
                      font=("Arial", 12),
                      bg="#e6f2ff", fg="#333")
emotion_label.pack(pady=5)

btn_style = {"font": ("Arial", 12),
             "bg": "#007acc",
             "fg": "white",
             "width": 22}

Button(root, text="Start Detection", command=start_detection, **btn_style).pack(pady=5)
Button(root, text="Stop Detection", command=stop_detection, **btn_style).pack(pady=5)
Button(root, text="Save Faces", command=save_detected_faces, **btn_style).pack(pady=5)
Button(root, text="Exit", command=close_app, **btn_style).pack(pady=10)

root.mainloop()
