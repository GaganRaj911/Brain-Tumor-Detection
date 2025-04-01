import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the GLCM features dataset
glcm_csv = "C:/GAGAN/Programming/Python ML/Features_files/glcm_features.csv"  # Update this if your file name is different
df = pd.read_csv(glcm_csv)

# Separate features (X) and labels (y)
X = df.iloc[:, 2:].values  # Select all columns except Filename and Label
y = df["Label"].values  # Label column (0 = non-tumor, 1 = tumor)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can change 'n_neighbors' as needed
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy on GLCM Features: {accuracy * 100:.2f}%")
