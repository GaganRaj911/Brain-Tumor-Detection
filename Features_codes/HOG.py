import os
import pandas as pd
from skimage.feature import hog
from skimage.io import imread

# Define folders (Change these to your actual paths)
tumor_folder = "C:/GAGAN/Programming/Python ML/brain_tumor_dataset/Resized yes"
non_tumor_folder = "C:/GAGAN/Programming/Python ML/brain_tumor_dataset/Resized no"

# HOG parameters
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

# Function to extract HOG features
def extract_hog(image):
    features = hog(image, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys')
    return features

# Process all images
data = []

for folder, label in [(tumor_folder, 1), (non_tumor_folder, 0)]:  # 1 = Tumor, 0 = No Tumor
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        image = imread(image_path, as_gray=True)  # Load grayscale image
        
        hog_features = extract_hog(image)
        data.append([filename, label] + hog_features.tolist())

# Convert to DataFrame
columns = ["Filename", "Label"] + [f"HOG_{i}" for i in range(len(hog_features))]
df = pd.DataFrame(data, columns=columns)

# Save to CSV (single file)
df.to_csv("hog_features.csv", index=False)
print("HOG features saved to hog_features.csv")
