import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # number of cores

# Load the feature datasets
hog_df = pd.read_csv("C:/GAGAN/Programming/Python ML/Features_files/hog_features.csv")
lbp_df = pd.read_csv("C:/GAGAN/Programming/Python ML/Features_files/lbp_features.csv")
# glcm_df = pd.read_csv("C:/GAGAN/Programming/Python ML/Features_files/glcm_features.csv")

# Merge datasets on 'Filename' and 'Label'
df = hog_df.merge(lbp_df, on=["Filename", "Label"])

# Separate features (X) and labels (y)
X = df.iloc[:, 2:].values  # All feature columns (skip Filename & Label)
y = df["Label"].values  # Labels (0 = no tumor, 1 = tumor)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy on Combined Features (HOG + LBP): {accuracy * 100:.2f}%")
