import os
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.io import imread

def extract_lbp_features(image, radius=3, n_points=8 * 3):
    """Extract LBP histogram features from an image"""
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit grayscale

    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    
    return hist


def process_folders(tumor_folder, non_tumor_folder, output_csv="lbp_features.csv"):
    """Extract LBP features from both tumor and non-tumor images and save to CSV"""
    feature_list = []
    
    # Process tumor images
    for filename in os.listdir(tumor_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(tumor_folder, filename)
            image = imread(image_path, as_gray=True)
            
            lbp_features = extract_lbp_features(image)
            feature_list.append([filename, 1] + list(lbp_features))  # Label = 1 for tumor
    
    # Process non-tumor images
    for filename in os.listdir(non_tumor_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(non_tumor_folder, filename)
            image = imread(image_path, as_gray=True)
            
            lbp_features = extract_lbp_features(image)
            feature_list.append([filename, 0] + list(lbp_features))  # Label = 0 for non-tumor
    
    # Convert to DataFrame
    columns = ["Filename", "Label"] + [f"LBP_{i}" for i in range(len(lbp_features))]
    df = pd.DataFrame(feature_list, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"LBP features saved to {output_csv}")

# Example usage
tumor_folder = "C:/GAGAN/Programming/Python ML/brain_tumor_dataset/Resized yes"  # tumor image folder path
non_tumor_folder = "C:/GAGAN/Programming/Python ML/brain_tumor_dataset/Resized no"  # non-tumor image folder path
process_folders(tumor_folder, non_tumor_folder)
