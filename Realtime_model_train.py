from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Realtime_model import create_object_detection_model

processed_path= "C:\\Users\\user\\Documents\\src\\src\\processed_realtime_dataset"
train_dir = os.path.join(processed_path, 'train')
valid_dir = os.path.join(processed_path, 'valid')
test_dir = os.path.join(processed_path, 'test')

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(256, 256), batch_size=32, class_mode='binary'
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir, target_size=(256, 256), batch_size=32, class_mode='binary'
)

#create the model
detection_model = create_object_detection_model(input_shape=(256, 256, 3))
detection_model.compile(
    optimizer='adam',
    loss={'bounding_box': 'mse', 'class': 'binary_crossentropy'},
    metrics={'bounding_box': 'mse', 'class': 'accuracy'}
)

#callbacks
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#training
history = detection_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=100,
    callbacks=[checkpoint, early_stop]
)
