# EYEDETECTION

Code Description - Line by Line in Points

Imports and Setup:

1.import streamlit as st: Imports Streamlit for creating a web-based application interface.

2.import cv2: Imports OpenCV for face and eye detection using pre-trained cascades.

3.import numpy as np: Imports NumPy for array manipulation and encoding storage.

4.import sqlite3: Imports SQLite3 for managing a lightweight local database.

5.from PIL import Image: Imports the PIL library to handle image input and processing.

6.import os: Allows interaction with the operating system for file paths.

7.import face_recognition: Imports the face_recognition library for facial encoding and comparison.

Database Initialization:

8.def init_db():: Defines a function to initialize the SQLite database.

9.sqlite3.connect('faces.db'): Connects to or creates a database file named faces.db.

10.c.execute(...): Creates a table known_faces with columns for ID, name, and encoding, if it doesn’t already exist.

11.conn.commit(): Commits the transaction to save changes to the database.

12.conn.close(): Closes the database connection.

