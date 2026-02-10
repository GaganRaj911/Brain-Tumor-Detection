import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Load Features and Labels ===
hog = np.load("hog_features.npy")
lbp = np.load("lbp_features.npy")
glcm = np.load("glcm_features.npy")
labels = np.load("labels.npy")

X = np.hstack((hog, lbp, glcm))
y = labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Define Hyperparameter Grid ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

# === Grid Search ===
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# === Best Model and Evaluation ===
best_rf = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTuned Random Forest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Tumor", "Tumor"])
disp.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()
