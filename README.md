# EYEDETECTION

Code Description - Line by Line in Points
Imports and Setup
import streamlit as st: Imports Streamlit for creating a web-based application interface.
import cv2: Imports OpenCV for face and eye detection using pre-trained cascades.
import numpy as np: Imports NumPy for array manipulation and encoding storage.
import sqlite3: Imports SQLite3 for managing a lightweight local database.
from PIL import Image: Imports the PIL library to handle image input and processing.
import os: Allows interaction with the operating system for file paths.
import face_recognition: Imports the face_recognition library for facial encoding and comparison.
Database Initialization
def init_db():: Defines a function to initialize the SQLite database.
sqlite3.connect('faces.db'): Connects to or creates a database file named faces.db.
c.execute(...): Creates a table known_faces with columns for ID, name, and encoding, if it doesn’t already exist.
conn.commit(): Commits the transaction to save changes to the database.
conn.close(): Closes the database connection.
Saving Face Data to Database
def save_face_to_db(name, encoding):: Function to save a face encoding to the database.
c.execute("INSERT INTO ... VALUES (?, ?)", ...): Inserts a name and binary encoding data into the database table.
encoding.tobytes(): Converts the NumPy array encoding to bytes for storage.
conn.commit() & conn.close(): Finalizes and closes the database connection.
Loading Known Faces from Database
def load_known_faces_from_db():: Function to load all face data stored in the database.
c.execute("SELECT * FROM known_faces"): Fetches all records from the known_faces table.
np.frombuffer(face[2], dtype=np.float64): Converts binary-encoded face data back into a NumPy array for use.
Returns two lists: One containing encodings (known_face_encodings) and the other with corresponding names (known_face_names).
Face and Eye Detection with OpenCV
def detect_faces_and_eyes(image):: Function to detect faces and eyes in an image.
cv2.CascadeClassifier(...): Loads pre-trained Haar cascades for face and eye detection.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY): Converts the image to grayscale for better cascade performance.
faces = face_cascade.detectMultiScale(...): Detects faces in the image and returns bounding box coordinates.
cv2.rectangle(...): Draws rectangles around detected faces.
Detects eyes: Searches for eyes within the detected face regions and marks them with rectangles.
Face Comparison with Database
def compare_faces(...):: Function to match detected face with stored encodings.
face_recognition.face_encodings(unknown_image): Extracts encodings of detected faces from the image.
face_recognition.compare_faces(...): Compares unknown encodings with known encodings from the database.
face_recognition.face_distance(...): Calculates similarity scores between the encodings.
np.argmin(face_distances): Finds the closest match based on minimum distance.
Returns matched name and accuracy: Outputs the name of the matched person and the confidence score.
Streamlit Application Layout
st.title(...): Sets the title of the Streamlit app.
st.selectbox(...): Allows users to select between two input methods: "Camera" or "Upload Image".
init_db(): Initializes the database at the start of the application.
load_known_faces_from_db(): Loads existing face data from the database.
Camera Input Handling
if option == "Camera":: Handles real-time input from the camera.
cv2.VideoCapture(0): Opens the default camera for capturing frames.
stframe = st.empty(): Creates a placeholder in the Streamlit app for displaying video frames.
Face and eye detection in real-time: Calls detect_faces_and_eyes() on each captured frame.
Face recognition in real-time: Matches faces in the frame against the database using compare_faces().
st.text_input(...): Prompts the user to input a name if an unknown face is detected.
Image Upload Handling
elif option == "Upload Image":: Handles image uploads.
st.file_uploader(...): Allows users to upload image files.
Processes uploaded image: Detects faces, eyes, and performs recognition using the same functions as in the camera input.
Saving New Faces
Unknown faces: If no match is found, extracts face encoding using face_recognition.face_encodings().
Prompts user for name: Saves the new face encoding with the provided name to the database.
Stopping the Application
if cv2.waitKey(1) & 0xFF == ord('q'):: Allows the user to stop the camera input by pressing the "q" key.
cam.release(): Releases the camera resource.
