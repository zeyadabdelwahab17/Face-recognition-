import os
import cv2
import time
import pickle
import tempfile
import numpy as np
import face_recognition
from gtts import gTTS
import pygame

# Path to your trained model file
MODEL_PATH = "face_recognition_model.pkl"

# Initialize pygame mixer
pygame.mixer.init()

def speak(text):
    print(f"Speaking: {text}")
    tts = gTTS(text=text, lang='en')
    temp_file = os.path.join(tempfile.gettempdir(), "feedback.mp3")
    tts.save(temp_file)

    sound = pygame.mixer.Sound(temp_file)
    sound.play()
    while pygame.mixer.get_busy():
        time.sleep(0.1)
    
    os.remove(temp_file)

def open_logitech_camera():
    # Try Logitech webcam (usually index 1 or 2 on Raspberry Pi)
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    speak("Error: Could not open any webcam.")
    return None

def recognize_once():
    if not os.path.exists(MODEL_PATH):
        speak("No trained model found.")
        return

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    cap = open_logitech_camera()
    if not cap:
        return

    time.sleep(2)  # Let camera warm up
    ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Failed to capture frame.")
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        speak("No face detected.")
        return

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            matched_indexes = [i for i, match in enumerate(matches) if match]
            name_counts = {}
            for i in matched_indexes:
                matched_name = data["names"][i]
                name_counts[matched_name] = name_counts.get(matched_name, 0) + 1
            name = max(name_counts, key=name_counts.get)

        speak(f"Detected: {name}")

if __name__ == "__main__":
    recognize_once()
