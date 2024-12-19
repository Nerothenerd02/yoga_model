from sklearn.svm import SVC
from utils import evaluate
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from joblib import dump, load

def predict_video(model, video_path, show=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract keypoints (x, y, z, visibility)
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            keypoints = np.array(keypoints).reshape(1, -1)

            # Validate the feature size
            if keypoints.shape[1] != 132:
                print(f"Warning: Expected 132 features, but got {keypoints.shape[1]}")
                continue

            # Predict pose
            prediction = model.predict(keypoints)[0]
            cv2.putText(frame, f"Pose: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if show:
            cv2.imshow('Pose Detection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Load data
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

# Clean data
if "Unnamed: 0" in data_train.columns:
    data_train.drop(labels="Unnamed: 0", axis=1, inplace=True)
if "Unnamed: 0" in data_test.columns:
    data_test.drop(labels="Unnamed: 0", axis=1, inplace=True)

# Train model
X, Y = data_train.iloc[:, :-1], data_train['target']
model = SVC(kernel='poly', decision_function_shape='ovo')
model.fit(X, Y)

# Save the model
dump(model, "svm_model.pkl")
print("Model saved as svm_model.pkl")

# Evaluate model
predictions = evaluate(data_test, model, show=True)

# Predict video
predict_video(model, "vid2.mp4", show=True)

predict_video(model, "vid1.mp4", show=True)