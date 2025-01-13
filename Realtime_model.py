import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from model import create_feature_extractor

def create_feature_extractor(input_shape=(256, 256, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the layers of ResNet50 during initial training
    return base_model

def create_object_detection_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    #feature extractor (ResNet backbone)
    feature_extractor = create_feature_extractor(input_shape)
    x = feature_extractor(inputs)

    #flatten and add dense layers for bounding box prediction
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)

    #predict bounding box coordinates (x_min, y_min, x_max, y_max)
    bounding_box_output = layers.Dense(4, activation='sigmoid', name='bounding_box')(x)

    #output the class prediction (binary classification: recyclable vs non-recyclable)
    class_output = layers.Dense(1, activation='sigmoid', name='class')(x)

    #build the model
    model = models.Model(inputs, [bounding_box_output, class_output])
    return model


def preprocess_frame(frame, target_size=(256, 256)):
    """Preprocess a frame for the model."""
    img = cv2.resize(frame, target_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

def draw_detections(frame, boxes, labels, scores):
    """Draw bounding boxes and labels on the frame."""
    for i, box in enumerate(boxes):
        if len(box) != 4:
            print(f"Skipping invalid box: {box}")
            continue

        label = labels[i]
        score = scores[i]

        x_min, y_min, x_max, y_max = box  #bounding box coordinates

        #draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        #display label and score
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

#non-max suppression
def non_max_suppression(boxes, scores, threshold=0.5):
    """Apply non-max suppression to filter overlapping boxes."""
    boxes = np.array(boxes)
    scores = np.array(scores)

    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        others = idxs[1:]

        if len(others) > 0:
            ious = compute_iou(boxes[current], boxes[others])
            idxs = others[ious < threshold]
        else:
            break

    return keep


def compute_iou(box1, boxes):
    """Compute IoU between box1 and multiple boxes."""
    if len(boxes.shape) == 1:  # If boxes is a single box
        boxes = np.expand_dims(boxes, axis=0)

    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box1_area + boxes_area - inter_area

    return inter_area / union_area


def real_time_detection(model, video_source=0, target_size=(256, 256)):
    """Perform real-time object detection and classification."""
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #preprocess the frame
        input_frame = preprocess_frame(frame, target_size)

        #get predictions
        bounding_boxes, class_scores = model.predict(input_frame)

        #scale bounding boxes to the original frame size
        h, w, _ = frame.shape
        bounding_boxes = bounding_boxes[0] * [w, h, w, h]  # Rescale to frame dimensions
        bounding_boxes = bounding_boxes.astype(np.int32)

        #apply non-max suppression
        scores = class_scores[0]
        nms_idxs = non_max_suppression(bounding_boxes, scores)

        #filter boxes, scores, and labels after NMS
        bounding_boxes = bounding_boxes[nms_idxs]
        if bounding_boxes.ndim == 1:  # Ensure 2D shape for a single box
            bounding_boxes = np.expand_dims(bounding_boxes, axis=0)

        scores = scores[nms_idxs]
        labels = ["Recyclable" if score > 0.5 else "Non-Recyclable" for score in scores]

        #print bounding boxes and their shapes
        print(f"Bounding boxes: {bounding_boxes}")
        print(f"Bounding boxes shape: {bounding_boxes.shape}")

        #print bounding boxes and their shapes
        print(f"Bounding boxes: {bounding_boxes}")
        print(f"Bounding boxes shape: {bounding_boxes.shape}")

        #draw detections on the frame
        frame = draw_detections(frame, bounding_boxes, labels, scores)

        #display the frame
        cv2.imshow("Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#example
if __name__ == "__main__":
    detection_model = create_object_detection_model(input_shape=(256, 256, 3))
    real_time_detection(detection_model)
