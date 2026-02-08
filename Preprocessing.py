import cv2
import os
import numpy as np

dataset_path="" # ← Path to your dataset 

def preprocess_images(dataset_path, image_size=(128, 128)):
    data = []
    labels = []

    for label_name in ['Tumor', 'No_Tumor']:
        label = 1 if label_name == 'Tumor' else 0
        folder = os.path.join(dataset_path, label_name)
        
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, image_size)
                img = img / 255.0  # Normalize to [0, 1]
                data.append(img)
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

data, labels = preprocess_images(dataset_path)

np.save("data.npy", data)
np.save("labels.npy", labels)

print("Done")
