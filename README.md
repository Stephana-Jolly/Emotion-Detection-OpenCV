# 🧠 Real-Time Emotion Detection using OpenCV & DeepFace
---
## 📘 Overview

This project performs real-time human emotion detection from live webcam feed using OpenCV for face detection and DeepFace for emotion analysis.

The system automatically detects faces in the video stream and classifies emotions such as:
😄 happy, 😐 neutral, 😡 angry, 😢 sad, 😲 surprise, 😴 fear, and 😌 disgust

---

## 🏗️ Tech Stack

Python 3.10+

OpenCV – for face detection and video capture

DeepFace – for emotion classification

TensorFlow / Keras – backend deep learning framework used by DeepFace

Haarcascade Classifier – for detecting faces in each frame

## ⚙️ How It Works

Face Detection:
Uses the pre-trained haarcascade_frontalface_default.xml classifier from OpenCV to detect faces in each video frame.

Emotion Analysis:
Each detected face is passed to DeepFace’s pretrained emotion recognition model, which predicts the emotion label with confidence scores.

Display:
The system draws a rectangle around each detected face and overlays the detected emotion on the live video feed.

## 💻 Project Workflow
Webcam → Frame Capture → Face Detection → Emotion Prediction → Display Result

## 📂 File Structure
Project_Emotion_opencv/
│
├── detection.py # Main script
├── haarcascade_frontalface_default.xml
├── README.md # (this file)
└── requirements.txt # Dependencies list (optional)

## 🚀 How to Run
1️⃣ Clone the repository
git clone https://github.com/<your-username>/Project_Emotion_opencv.git
cd Project_Emotion_opencv

2️⃣ Install dependencies

Make sure you have Python and pip installed.
Then install all required libraries:

pip install opencv-python deepface

3️⃣ Run the script

python detection.py

4️⃣ Quit

Press ‘q’ to close the webcam window.


