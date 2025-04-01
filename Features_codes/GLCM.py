import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread

def extract_glcm_features(image, distances=[1], angles=[0]):
    """Extract GLCM texture features (contrast, correlation, energy, homogeneity) from an image"""
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit grayscale

    # Compute GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Extract texture features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return [contrast, correlation, energy, homogeneity]

def process_folders(tumor_folder, non_tumor_folder, output_csv="glcm_features.csv"):
    """Extract GLCM features from both tumor and non-tumor images and save to CSV"""
    feature_list = []
    
    # Process tumor images
    for filename in os.listdir(tumor_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(tumor_folder, filename)
            image = imread(image_path, as_gray=True)

            glcm_features = extract_glcm_features(image)
            feature_list.append([filename, 1] + glcm_features)  # Label = 1 for tumor
    
    # Process non-tumor images
    for filename in os.listdir(non_tumor_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(non_tumor_folder, filename)
            image = imread(image_path, as_gray=True)

            glcm_features = extract_glcm_features(image)
            feature_list.append([filename, 0] + glcm_features)  # Label = 0 for non-tumor
    
    # Convert to DataFrame
    columns = ["Filename", "Label", "Contrast", "Correlation", "Energy", "Homogeneity"]
    df = pd.DataFrame(feature_list, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"GLCM features saved to {output_csv}")

# Example usage
tumor_folder = "C:/GAGAN/Programming/Python ML/brain_tumor_dataset/Resized yes"  #  tumor image folder path
non_tumor_folder = "C:/GAGAN/Programming/Python ML/brain_tumor_dataset/Resized no"  #  non-tumor image folder path
process_folders(tumor_folder, non_tumor_folder)
