# model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Choose a pretrained feature extractor (ResNet as the backbone)
def create_feature_extractor(input_shape=(256, 256, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the layers of ResNet50 during initial training
    return base_model

# 3.2 Build an object detection model from scratch (using bounding boxes)
# Note: For object detection, we would typically require bounding box annotations.
# We will create a simplified model to demonstrate localization using bounding boxes.
def create_object_detection_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Feature extractor (ResNet backbone)
    feature_extractor = create_feature_extractor(input_shape)
    x = feature_extractor(inputs)
    
    # Flatten and add dense layers for bounding box prediction
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # Predict bounding box coordinates (x_min, y_min, x_max, y_max)
    bounding_box_output = layers.Dense(4, activation='sigmoid', name='bounding_box')(x)
    
    # Output the class prediction (binary classification: recyclable vs non-recyclable)
    class_output = layers.Dense(1, activation='sigmoid', name='class')(x)
    
    # Build the model
    model = models.Model(inputs, [bounding_box_output, class_output])
    return model

# 3.3 Develop the classification model:
# Build a CNN architecture and fine-tune the model on our dataset
def create_classification_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Feature extractor (ResNet backbone)
    feature_extractor = create_feature_extractor(input_shape)
    x = feature_extractor(inputs)
    
    # Flatten and add dense layers for classification
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # Output layer for binary classification (Recyclable vs Non-Recyclable)
    class_output = layers.Dense(1, activation='sigmoid', name='class')(x)
    
    # Build the model
    model = models.Model(inputs, class_output)
    return model

# Data Augmentation (used in model training if necessary)
def get_data_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen

# Compile the models
def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={'bounding_box': 'mean_squared_error', 'class': 'binary_crossentropy'},
        metrics={'bounding_box': 'accuracy', 'class': 'accuracy'}
    )

# Example of how to use these models:
if __name__ == "__main__":
    # Classification model
    model = create_classification_model(input_shape=(256, 256, 3))
    model.summary()  # Print the model summary to check architecture
    
    # Object Detection model
    detection_model = create_object_detection_model(input_shape=(256, 256, 3))
    detection_model.summary()  # Print the model summary for object detection model
