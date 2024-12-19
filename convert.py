import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
import joblib

# Load your SVM model
model = joblib.load("svm_model.pkl")

# Create a TensorFlow model
def svm_to_tensorflow(features):
    prediction = model.predict(features)
    return np.array(prediction, dtype=np.float32)

# Convert to TFLite
concrete_func = tf.function(svm_to_tensorflow).get_concrete_function(tf.TensorSpec([None, 132], tf.float32))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TFLite model
with open("pose_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion to TensorFlow Lite format completed successfully.")