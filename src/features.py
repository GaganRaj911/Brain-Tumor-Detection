import numpy as np
from skimage.feature import hog

def extract_hog_features(data, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    hog_features = []

    for img in data:
        features = hog(img,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm='L2-Hys')
        hog_features.append(features)

    return np.array(hog_features)

if __name__=='__main__':
    data = np.load("data.npy")   

    hog_features = extract_hog_features(data)

    print("HOG feature shape:", hog_features.shape)

    np.save("hog_features.npy", hog_features)

    print("Done")