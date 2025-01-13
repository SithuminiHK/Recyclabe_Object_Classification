import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Realtime_model import create_feature_extractor
from Realtime_model import draw_detections

#load the trained model
detection_model = load_model('best_model.h5', custom_objects={"create_feature_extractor": create_feature_extractor})

#start video capture
cap = cv2.VideoCapture(0)

def preprocess_image(image_array, target_size=(256, 256)):
    """Preprocess a NumPy array image for the model."""
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    #resize the image
    img = cv2.resize(image_array, target_size)
    img = img / 255.0  #normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)
    return img

while True:
    #capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    #preprocess the frame
    preprocessed_image = preprocess_image(frame, target_size=(256, 256))

    #get predictions from the model
    bounding_boxes, class_scores = detection_model.predict(preprocessed_image)

    #rescale bounding boxes to the original frame size
    h, w, _ = frame.shape
    bounding_boxes = bounding_boxes[0] * [w, h, w, h]
    bounding_boxes = bounding_boxes.astype(int)

    #ensure bounding_boxes is 2D
    bounding_boxes = bounding_boxes.reshape(-1, 4)

    #labels based on class scores
    labels = ["Recyclable" if score > 0.5 else "Non-Recyclable" for score in class_scores[0]]

    #draw bounding boxes and labels on the frame
    output_frame = draw_detections(frame, bounding_boxes, labels, class_scores[0])

    #display the resulting frame
    cv2.imshow("Real-Time Detection", output_frame)

    #press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
