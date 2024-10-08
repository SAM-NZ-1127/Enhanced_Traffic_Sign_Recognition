# Enhanced_Traffic_Sign_Recognition

This project aims to develop a Convolutional Neural Network (CNN) model to recognize and classify traffic signs. We enhance the model's accuracy by utilizing various image processing techniques for pre-processing the images. The project is part of an effort to improve traffic safety and facilitate automated transportation systems.

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Steps](#steps)
  - [Step 1: Image Preprocessing](#step-1-image-preprocessing)
  - [Step 2: Image Filtering](#step-2-image-filtering)
  - [Step 3: Model Training](#step-3-model-training)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Applications](#applications)
- [Contributors](#contributors)

## Motivation

Traffic signs are essential for managing traffic flow and ensuring road safety. This project leverages CNNs to recognize traffic signs accurately. By utilizing image processing techniques, we aim to further enhance the accuracy and reliability of the model, which is crucial for intelligent transportation systems and autonomous vehicles.

## Dataset

We used a subset of the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset to develop and test the model.

## Technologies Used

- Python
- TensorFlow
- OpenCV
- Scikit-learn

## Steps

### Step 1: Image Preprocessing
- **Conversion to Grayscale**: Simplifies images by converting them from three color channels to one.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances image contrast for better visibility of important features.

### Step 2: Image Filtering
- **Gaussian Filter**: Reduces noise and details in the image.
- **Sobel Filter**: Enhances edge detection.
- **Sharpening Filter**: Improves the edges in the image.
- **Normalization**: Scales pixel values to a consistent range to improve training convergence.

### Step 3: Model Training
- **Image Augmentation**: Introduces variability in the dataset through random rotations and scaling to improve the robustness of the model.
- **CNN Architecture**:
  - Convolutional Layers: Extract features from input images.
  - Pooling Layers: Reduce spatial dimensions and complexity.
  - Dropout: Prevents overfitting.
  - Output Layer: Multi-class classification of traffic signs.

## Model Architecture

The CNN architecture includes multiple convolutional layers, pooling layers, and a fully connected dense output layer for classification. We introduced various filters in our image preprocessing pipeline to enhance accuracy further.

## Results

- **Without Filters**: The model achieved 93% accuracy after 25 epochs.
- **With Filters**: The model improved to 96% accuracy after 25 epochs.

## Applications

- **Autonomous Vehicles**: Traffic sign recognition for safer navigation and law compliance.
- **Driver Assistance Systems**: Enhances systems like adaptive cruise control and speed alert systems.
- **Traffic Law Enforcement**: Monitoring compliance with traffic laws.
- **Educational Tools**: Used in driving simulators for training new drivers.
- **Vision Disability Aid**: Assists visually impaired individuals in recognizing traffic signs for safer commutes.

## Contributors

- **Sai Keerthi Nelluri**: Image Filtering
- **Shivam Jayeshkumar Mehta**: Model Training
- **Rajesh Mahendran**: Image Preprocessing

---

This project demonstrates the use of CNN and image processing to create an efficient traffic sign recognition system, achieving high accuracy through the GTSRB dataset. For future improvements, we plan to account for a wider range of weather conditions and larger datasets to further enhance the model's robustness.
