import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from model import create_feature_extractor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Initialize the ImageDataGenerator for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Define the test data generator
test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\user\\Documents\\src\\src\\processed_realtime_dataset\\test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load the model
model = load_model('best_model.h5', custom_objects={"create_feature_extractor": create_feature_extractor})

# Evaluate the model on the test dataset
results = model.evaluate(test_generator)

# Unpack the results
total_loss, bounding_box_loss, class_loss, bounding_box_mse, class_accuracy = results

print(f"Total Loss: {total_loss}")
print(f"Bounding Box Loss: {bounding_box_loss}")
print(f"Class Loss: {class_loss}")
print(f"Bounding Box MSE: {bounding_box_mse}")
print(f"Class Accuracy: {class_accuracy}")

# Get all true labels from the test generator
true_labels = test_generator.classes  # Binary class labels for all samples

# Predict for the entire test set
predictions = model.predict(test_generator, verbose=1)
predicted_labels = (predictions[1] > 0.4).astype(int).flatten()

# Print shapes for debugging
print("True Labels Shape:", true_labels.shape)
print("Predicted Labels Shape:", predicted_labels.shape)

# Evaluate metrics
#evaluate_metrics(true_labels, predicted_labels)

# Get a batch of images and labels

# Function to evaluate precision, recall, F1-score, and confusion matrix
def evaluate_metrics(true_labels, predicted_labels):
    """Evaluate and print precision, recall, F1-score, and confusion matrix."""
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Recyclable', 'Recyclable'])
    plt.yticks(tick_marks, ['Non-Recyclable', 'Recyclable'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Evaluate the metrics on the current batch
evaluate_metrics(true_labels, predicted_labels)
