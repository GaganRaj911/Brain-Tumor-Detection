import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set this to your actual number of cores

# Load HOG features dataset
df = pd.read_csv("C:\GAGAN\Programming\Python ML\Features_files\hog_features.csv")

# Extract features and labels
X = df.iloc[:, 2:].values  # All HOG feature columns
y = df["Label"].values     # Labels (0 = No Tumor, 1 = Tumor)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train KNN model with k=5 (you can experiment with different values)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy on HOG Features: {accuracy * 100:.2f}%")
