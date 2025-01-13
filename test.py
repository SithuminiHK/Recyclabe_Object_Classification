import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from model import create_feature_extractor
from Realtime_model import draw_detections

# Load the trained model
detection_model = load_model('best_model.h5', custom_objects={"create_feature_extractor": create_feature_extractor})

# Load a sample image
image_path = 'C:\\Users\\user\\Documents\\src\\src\\processed_realtime_dataset\\test\\Recyclable\\Image_7.png'
frame = cv2.imread(image_path)

def preprocess_image(image_array, target_size=(256, 256)):
    """Preprocess a NumPy array image for the model."""
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Resize the image
    img = cv2.resize(image_array, target_size)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Preprocess the image
preprocessed_image = preprocess_image(frame, target_size=(256, 256))

# Get predictions
bounding_boxes, class_scores = detection_model.predict(preprocessed_image)

# Rescale bounding boxes to the original frame size
h, w, _ = frame.shape
bounding_boxes = bounding_boxes[0] * [w, h, w, h]
bounding_boxes = bounding_boxes.astype(int)

# Ensure bounding_boxes is 2D
bounding_boxes = bounding_boxes.reshape(-1, 4)  # Reshape to 2D if it's flat

print("Bounding Boxes:", bounding_boxes)
print("Type:", type(bounding_boxes))
print("Shape:", getattr(bounding_boxes, "shape", None))

# Draw predictions on the image
labels = ["Recyclable" if score > 0.5 else "Non-Recyclable" for score in class_scores[0]]
output_frame = draw_detections(frame, bounding_boxes, labels, class_scores[0])

# Show the output
cv2.imshow("Test Image", output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
