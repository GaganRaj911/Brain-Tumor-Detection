# Brain Tumor Detection using HOG and Ensemble Machine Learning

## Overview
This project focuses on detecting brain tumors from MRI images using **classical machine learning techniques**. Instead of deep learning models, the system uses **Histogram of Oriented Gradients (HOG)** for feature extraction and an **ensemble of machine learning classifiers** for efficient and interpretable classification.

The task is a **binary classification** problem:
- Tumor
- No Tumor

This approach is computationally lightweight and suitable for systems with limited resources.

---

## Dataset
- Total images: **3120**
  - Tumor: **1550**
  - No Tumor: **1570**
- Image type: Brain MRI scans
- Image format: Grayscale
- Image size after preprocessing: **128 × 128**

> ⚠️ The dataset is **not included** in this repository.  
> Please refer to the project report for dataset source and details.

---

## Methodology

### 1. Image Preprocessing
- Converted images to grayscale
- Resized images to **128 × 128**
- Normalized pixel values to the range **[0, 1]**
- Saved processed images and labels as NumPy arrays

---

### 2. Feature Extraction
- Used **Histogram of Oriented Gradients (HOG)**
- Parameters:
  - Cell size: 8 × 8
  - Block size: 2 × 2
  - Orientation bins: 9
- Final feature vector size: ~8100 features per image

HOG captures structural and edge information critical for distinguishing tumor regions in MRI scans.

---

### 3. Machine Learning Models
The following classifiers were trained and evaluated:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
  - Linear kernel
  - Polynomial kernel
  - RBF kernel
- Random Forest Classifier

---

### 4. Ensemble Model
An ensemble classifier was created using **majority voting** with:
- KNN
- Polynomial SVM
- RBF SVM

The ensemble achieved the best overall performance.

---

## Results

| Model               | Accuracy |
|--------------------|----------|
| KNN                | 96.47%   |
| Linear SVM         | 95.35%   |
| Polynomial SVM     | 96.79%   |
| RBF SVM            | 96.96%   |
| Random Forest      | 90.87%   |
| **Ensemble Model** | **97.44%** |

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Project Structure
```text
brain-tumor-detection/
│
├── src/
│ ├── preprocess.py           # Image preprocessing
│ ├── features.py        # HOG feature extraction
│ ├── predict.py              # Single image prediction
│ └── Algorithms/
│   ├── knn.py 
│   ├── RandomForest.py
│   ├── linear_svm.py
│   ├── poly_svm.py
│   ├── rbf_svm.py
│   └── Tests/                # Hyperparameter tuning experiments
│     ├── knntest.py
│     ├── poly_svm_test.py
│     ├── rbf_svm_test.py
│     └── Rf_test.py
│ 
├── media/
│ └── working_demo.mp4
│
├── project_report.pdf        # Final project report
├── requirements.txt
└── README.md
```


## How to Run

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Preprocess Images
```bash
python src/preprocess.py
```

4. Extract HOG Features
```bash
python src/features.py
```

5. Train and Evaluate Models
Each algorithm file inside src/Algorithms/ trains and evaluates a specific classifier.
Run the required scripts to train models and compare their performance on your dataset.

6. Predict on a New Image
```bash
python src/predict.py --image path_to_image
```

## Trained Models
Trained model files are not included in this repository due to their large size (~300 MB).

Models are trained locally by running the scripts inside src/Algorithms/ after preprocessing and feature extraction.

## Requirements
All required Python dependencies for this project are listed in `requirements.txt`.

## Applications
Computer-aided diagnosis
Medical image analysis
Healthcare decision-support systems

## Limitations
Binary classification only
No tumor localization or segmentation
Performance depends on MRI quality and dataset diversity

## Future Improvements
Multi-class tumor classification
Tumor segmentation
Comparison with deep learning models

## Authors
I. Gagan
Team Members
Under the guidance of
B. Manasa, Assistant Professor, ECE
Jawaharlal Nehru Technological University Hyderabad
