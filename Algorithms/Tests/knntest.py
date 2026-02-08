import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Features and Labels
hog_features = np.load("Features/hog_features.npy")
labels = np.load("Dataset_Data/labels.npy")

# Assign variables to Features and Labels
X = hog_features
y = labels

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Evaluate Accuracy for Different k Values 
k_values = list(range(1, 20, 2))  # Odd values from 1 to 19
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k = {k}, Accuracy = {acc * 100:.2f}%")

# Plot Accuracy vs. k 
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title("KNN Accuracy vs. Number of Neighbors (k)")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(k_values)
plt.ylim(min(accuracies) - 0.02, 1.0)
plt.show()
