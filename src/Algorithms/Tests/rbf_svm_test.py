import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Load and Combine Features ===
hog = np.load("hog_features.npy")
lbp = np.load("lbp_features.npy")
glcm = np.load("glcm_features.npy")
labels = np.load("labels.npy")

X = np.hstack((hog, lbp, glcm))
y = labels

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Define Hyperparameter Grid for RBF SVM ===
param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 0.01, 0.001, 0.0001]
}

# === Grid Search with Cross-Validation ===
grid = GridSearchCV(
    SVC(kernel='rbf'), 
    param_grid, 
    cv=3, 
    n_jobs=-1, 
    verbose=1
)

grid.fit(X_train, y_train)

# === Best Model ===
best_svm_rbf = grid.best_estimator_
print("\nBest Parameters:", grid.best_params_)

# === Evaluation ===
y_pred = best_svm_rbf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nRBF SVM Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Tumor", "Tumor"])
disp.plot(cmap=plt.cm.Purples)
plt.title("Confusion Matrix - SVM (RBF Kernel)")
plt.show()
