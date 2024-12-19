import mediapipe as mp
import cv2
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report

# Build the dataset using landmarks detection and save it as csv
def build_dataset(path, dataset_type):
    """
    Build the dataset by extracting pose landmarks and saving them to a CSV file.
    
    Args:
    - path: str, Root path to the dataset.
    - dataset_type: str, Type of dataset (e.g., "train" or "test").
    
    Returns:
    - None, Saves the dataset to a CSV file.
    """
    # Check if the path exists
    if not os.path.exists(path):
        print(f"Error: Path does not exist - {path}")
        return

    # Initialize the DataFrame
    data_columns = []
    for p in points:
        x = str(p)[13:]
        data_columns.extend([f"{x}_x", f"{x}_y", f"{x}_z", f"{x}_vis"])
    data_columns.append("target")  # name of the position
    data = pd.DataFrame(columns=data_columns)

    count = 0

    # Collect subdirectories (pose classes)
    dirnames = [x[1] for x in os.walk(path)][0]
    if not dirnames:
        print(f"Error: No subdirectories found in {path}")
        return

    print(f"Processing dataset at: {path}")
    print(f"Found classes: {dirnames}")

    # Iterate through each pose class and process images
    for pose_class in dirnames:
        pose_path = os.path.join(path, pose_class)
        for img_name in os.listdir(pose_path):
            img_path = os.path.join(pose_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Unable to read image - {img_path}")
                continue

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                temp = [attr for j in landmarks for attr in [j.x, j.y, j.z, j.visibility]]
                temp.append(pose_class)  # Add the target class
                data.loc[count] = temp
                count += 1

    # Save the dataset to a CSV file
    csv_file = f"{dataset_type}.csv"
    data.to_csv(csv_file, index=False)
    print(f"Dataset saved to: {csv_file}")


# Predict the name of the poses in the image
def predict(img, model, show=False):
    temp = []
    img = cv2.imread(img)
    if img is None:
        print(f"Error: Unable to read image - {img}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for j in landmarks:
            temp.extend([j.x, j.y, j.z, j.visibility])
        y = model.predict([temp])

        if show:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            cv2.putText(img, str(y[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            cv2.imshow("image", img)
            cv2.waitKey(0)


# Evaluate the model
def evaluate(data_test, model, show=False):
    target = data_test.loc[:, "target"]  # list of labels
    target = target.values.tolist()
    predictions = []
    for i in range(len(data_test)):
        tmp = data_test.iloc[i, 0:len(data_test.columns) - 1]
        tmp = tmp.values.tolist()
        predictions.append(model.predict([tmp])[0])
    if show:
        print(confusion_matrix(predictions, target), '\n')
        print(classification_report(predictions, target))
    return predictions


# Mediapipe setup
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # For drawing keypoints
points = mpPose.PoseLandmark  # Landmarks

# Build train and test datasets
build_dataset("D:/SCHOOL WORK/3RD YEAR/FYP/Models/archive/DATASET/TRAIN", "train")
build_dataset("D:/SCHOOL WORK/3RD YEAR/FYP/Models/archive/DATASET/TEST", "test")

