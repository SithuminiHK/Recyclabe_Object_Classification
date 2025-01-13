import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model import create_feature_extractor
from Realtime_model import draw_detections

# Load the trained model
detection_model = load_model('best_model.h5', custom_objects={"create_feature_extractor": create_feature_extractor})

# Start video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

def preprocess_image(image_array, target_size=(256, 256)):
    """Preprocess a NumPy array image for the model."""
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Resize the image
    img = cv2.resize(image_array, target_size)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame
    preprocessed_image = preprocess_image(frame, target_size=(256, 256))

    # Get predictions from the model
    bounding_boxes, class_scores = detection_model.predict(preprocessed_image)

    # Rescale bounding boxes to the original frame size
    h, w, _ = frame.shape
    bounding_boxes = bounding_boxes[0] * [w, h, w, h]
    bounding_boxes = bounding_boxes.astype(int)

    # Ensure bounding_boxes is 2D
    bounding_boxes = bounding_boxes.reshape(-1, 4)

    # Labels based on class scores
    labels = ["Recyclable" if score > 0.5 else "Non-Recyclable" for score in class_scores[0]]

    # Draw bounding boxes and labels on the frame
    output_frame = draw_detections(frame, bounding_boxes, labels, class_scores[0])

    # Display the resulting frame
    cv2.imshow("Real-Time Detection", output_frame)

    # Press 'q' to exit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
