import cv2
from deepface import DeepFace

# Path to the haarcascade for face detection
face_cascade_path = r"D:\Data Science With AI\Vs code\Project_Emotion_opencv\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = frame[y:y + h, x:x + w]
        
        # Analyze emotion using DeepFace
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        
        # Get the dominant emotion
        emotion = result[0]['dominant_emotion']
        
        # Draw a rectangle around the face and put the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 225), 2)
    
    # Display the frame with emotion label
    cv2.imshow('Emotion Detection from Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
