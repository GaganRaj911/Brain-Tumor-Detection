import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# === Define Grid for Polynomial Kernel SVM ===
param_grid = {
    'C': [0.1, 1, 10],
    'degree': [2, 3],
    'gamma': ['scale'],  # or add 'auto'
    'coef0': [0, 1]       # coef0 controls influence of high vs low degree terms
}

poly_svm = SVC(kernel='poly')

grid = GridSearchCV(poly_svm, param_grid, cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# === Evaluate ===
best_poly = grid.best_estimator_
print("\nBest Polynomial SVM Parameters:", grid.best_params_)

y_pred = best_poly.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nPolynomial SVM Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

# === Plot Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Tumor", "Tumor"])
disp.plot(cmap="Blues")
plt.title("Polynomial SVM - Confusion Matrix")
plt.show()
