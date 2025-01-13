# Real time Recyclabe_Object_Classification
# Object Detection and Classification Model 
This repository contains the implementation and evaluation code for a Real-time object detection and classification model for real time. 

The project aims to develop an automated system using object detection and classification to distinguish between recyclable and non-recyclable household waste. This system will assist in effective waste management by detecting various objects and classifying them into appropriate categories, thereby promoting recycling efforts.

**Dataset:**
We used the "Recyclable and Household Waste Classification" dataset from Kaggle, which contains 15,000 images (each 256x256 pixels) depicting various recyclable materials, general waste, and household items across 30 distinct categories.
Link : https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification

**Folder Structure:**
data/processed_realtime_dataset/: Preprocessed dataset ready for model training and evaluation.
data/README.md: Includes the dataset description and download link.
src/: Contains Python scripts for various stages of the project:
  preprocessing.py: preprocessing raw data for training and testing.
  Realtime_model.py : defines model architecture for real time object detection
  Realtime_model_train.py: handles the training pipeline.
  evaluate_realtime.py: Evaluates model performance.
Results/: Contains training results, graphs, and logs.
