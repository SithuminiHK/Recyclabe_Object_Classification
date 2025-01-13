import kagglehub
import os
import random
import shutil
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("alistairking/recyclable-and-household-waste-classification")

print("Path to dataset files:", path)

# Define dataset paths
raw_dataset_path = '/root/.cache/kagglehub/datasets/alistairking/recyclable-and-household-waste-classification/versions/1'
dataset_path = os.path.join(raw_dataset_path, 'images', 'images')
processed_path = '/content/processed_realtime_dataset'

dataset_path = '/root/.cache/kagglehub/datasets/alistairking/recyclable-and-household-waste-classification/versions/1/images/images'
print("Contents of dataset path:", os.listdir(dataset_path))
for category in os.listdir(dataset_path):
    print(f"Contents of category {category}:", os.listdir(os.path.join(dataset_path, category)))

# Create directories for processed images
train_dir = os.path.join(processed_path, 'train')
valid_dir = os.path.join(processed_path, 'valid')
test_dir = os.path.join(processed_path, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to split dataset into train, validation, and test
def split_dataset(category, images):
    random.shuffle(images)
    num_images = len(images)
    num_train = int(0.7 * num_images)
    num_valid = int(0.15 * num_images)
    num_test = num_images - num_train - num_valid

    return (
        images[:num_train],
        images[num_train:num_train + num_valid],
        images[num_train + num_valid:]
    )

# Relabel categories into Recyclable and Non-Recyclable
def relabel_category(category):
    recyclable = [
        'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
        'cardboard_boxes', 'cardboard_packaging', 'glass_beverage_bottles',
        'glass_cosmetic_containers', 'glass_food_jars', 'magazines',
        'newspaper', 'office_paper', 'plastic_detergent_bottles',
        'plastic_food_containers', 'plastic_soda_bottles',
        'plastic_water_bottles', 'steel_food_cans'
    ]
    return 'Recyclable' if category in recyclable else 'Non-Recyclable'

# Process the dataset
categories = os.listdir(dataset_path)
all_images = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    default_path = os.path.join(category_path, 'real_world')

    if not os.path.isdir(default_path):
        continue

    relabeled_category = relabel_category(category)
    images = [img for img in os.listdir(default_path) if os.path.isfile(os.path.join(default_path, img))]
    all_images.extend([relabeled_category] * len(images))

    train_images, valid_images, test_images = split_dataset(relabeled_category, images)

    # Create subfolder directories in train, valid, and test
    os.makedirs(os.path.join(train_dir, relabeled_category), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, relabeled_category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, relabeled_category), exist_ok=True)

    # Copy images to respective directories
    for image in train_images:
        shutil.copy(os.path.join(default_path, image), os.path.join(train_dir, relabeled_category, image))
    for image in valid_images:
        shutil.copy(os.path.join(default_path, image), os.path.join(valid_dir, relabeled_category, image))
    for image in test_images:
        shutil.copy(os.path.join(default_path, image), os.path.join(test_dir, relabeled_category, image))

# Check class distribution
def check_class_distribution(categories):
    distribution = Counter(categories)
    plt.bar(distribution.keys(), distribution.values())
    plt.title("Class Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.show()

# Check and plot class distribution
check_class_distribution(all_images)

print("Dataset preparation complete: Split, Preprocessed, and Relabeled!")
