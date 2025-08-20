import cv2
import numpy as np
import winsound


# Initialize the face detector and landmark model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_model = cv2.face.createFacemarkLBF()
landmark_model.loadModel(r"C:\\Users\\User\\Documents\\6th Sem\\mini project\\Real\\lbfmodel.yaml")

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Status counters and settings
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
# Calculate the distance between two points
def compute(ptA, ptB):
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

# Determine if the eyes are closed, partially closed, or open
def blinked(landmarks):
    if len(landmarks) < 68:
        return 0.0  # Return a default value if landmarks are insufficient
    
    # Left eye landmarks
    left_eye = landmarks[36:42]
    # Right eye landmarks
    right_eye = landmarks[42:48]
    
    # Compute EAR for both eyes
    left_ear = (compute(left_eye[1], left_eye[5]) + compute(left_eye[2], left_eye[4])) / (2.0 * compute(left_eye[0], left_eye[3]))
    right_ear = (compute(right_eye[1], right_eye[5]) + compute(right_eye[2], right_eye[4])) / (2.0 * compute(right_eye[0], right_eye[3]))
    
    return (left_ear + right_ear) / 2.0

# EAR thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
            face_rect = (x, y, w, h)
            
            _, landmarks = landmark_model.fit(frame, np.array([face_rect]))
            if landmarks:
                landmarks = landmarks[0][0]  # Extract the first set of landmarks
                
                ear = blinked(landmarks)
                if ear < EAR_THRESHOLD:
                    sleep += 1
                    drowsy = 0
                    active = 0
                    if sleep >= EAR_CONSEC_FRAMES:
                        status = "SLEEPING !!!"
                        color = (255, 0, 0)
                        winsound.Beep(2000, 1000)  # Beep sound (frequency, duration in ms)
                elif EAR_THRESHOLD <= ear < EAR_THRESHOLD + 0.04:
                    drowsy += 1
                    sleep = 0
                    active = 0
                    if drowsy >= EAR_CONSEC_FRAMES:
                        status = "Drowsy !"
                        color = (0, 0, 255)
                        winsound.Beep(1500, 1000)  # Beep sound (frequency, duration in ms)
                else:
                    active += 1
                    sleep = 0
                    drowsy = 0
                    if active >= EAR_CONSEC_FRAMES:
                        status = "Active :)"
                        color = (0, 255, 0)
                
                cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                # Draw landmarks
                for point in landmarks:
                    px, py = int(point[0]), int(point[1])
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()