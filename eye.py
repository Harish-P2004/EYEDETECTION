import streamlit as st
import cv2
import numpy as np
import sqlite3
from PIL import Image
import os
import face_recognition

# Database setup
def init_db():
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS known_faces 
                 (id INTEGER PRIMARY KEY, name TEXT, encoding BLOB)''')
    conn.commit()
    conn.close()

# Function to save face data to database
def save_face_to_db(name, encoding):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute("INSERT INTO known_faces (name, encoding) VALUES (?, ?)", 
              (name, encoding.tobytes()))
    conn.commit()
    conn.close()

# Load known faces from the database
def load_known_faces_from_db():
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute("SELECT * FROM known_faces")
    known_faces = c.fetchall()
    conn.close()

    known_face_encodings = []
    known_face_names = []
    
    for face in known_faces:
        known_face_names.append(face[1])
        known_face_encodings.append(np.frombuffer(face[2], dtype=np.float64))
    
    return known_face_encodings, known_face_names

# Detect face and eyes using OpenCV
def detect_faces_and_eyes(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return image, faces

# Compare detected face with faces in the database
def compare_faces(unknown_image, known_face_encodings, known_face_names):
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)
    if len(unknown_face_encodings) > 0:
        unknown_encoding = unknown_face_encodings[0]
        results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)

        best_match_index = np.argmin(face_distances)
        accuracy = (1 - face_distances[best_match_index]) * 100

        if results[best_match_index]:
            return known_face_names[best_match_index], accuracy
    return None, 0

# Streamlit app layout
st.title("Eye Detection, Recognition, and Database Integration")
option = st.selectbox("Choose input method", ("Camera", "Upload Image"))

# Initialize the database
init_db()

# Load known faces from the database
known_face_encodings, known_face_names = load_known_faces_from_db()

# Use camera input
if option == "Camera":
    st.write("Click below to open camera:")
    if st.button("Open Camera"):
        cam = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cam.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            frame_with_faces, faces = detect_faces_and_eyes(frame)
            stframe.image(frame_with_faces, channels="BGR")

            if len(faces) > 0:
                person_name, accuracy = compare_faces(frame, known_face_encodings, known_face_names)
                if person_name:
                    st.write(f"Person: {person_name}, Accuracy: {accuracy:.2f}%")
                else:
                    st.write("No match found, saving new face to database...")
                    # Extract new face encoding and save it to the database
                    face_encodings = face_recognition.face_encodings(frame)
                    if face_encodings:
                        new_face_encoding = face_encodings[0]
                        name = st.text_input("Enter the name for this face")
                        if name:
                            save_face_to_db(name, new_face_encoding)
                            st.write(f"New face data saved for {name}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()

# Use image upload option
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        frame_with_faces, faces = detect_faces_and_eyes(image)
        st.image(frame_with_faces, caption="Uploaded Image", use_column_width=True)

        if len(faces) > 0:
            person_name, accuracy = compare_faces(image, known_face_encodings, known_face_names)
            if person_name:
                st.write(f"Person: {person_name}, Accuracy: {accuracy:.2f}%")
            else:
                st.write("No match found, saving new face to database...")
                # Extract new face encoding and save it to the database
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    new_face_encoding = face_encodings[0]
                    name = st.text_input("Enter the name for this face")
                    if name:
                        save_face_to_db(name, new_face_encoding)
                        st.write(f"New face data saved for {name}")