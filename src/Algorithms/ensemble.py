import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Make an ensemble of the Top performing models
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_poly= SVC(kernel='poly', C=1, gamma='scale',coef0=1,degree=4, probability=True)
knn = KNeighborsClassifier(n_neighbors=1)

ensemble = VotingClassifier(
    estimators=[
        ('svm1', svm_rbf),
        ('knn', knn),
        ('svm2',svm_poly)
    ],
    voting='soft'  
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Voting Ensemble Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Tumor", "Tumor"])
disp.plot(cmap="Blues")
plt.title("Final Ensemble - Confusion Matrix")
plt.show()

# 
import joblib
joblib.dump(ensemble, "brain_tumor_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print("Model and scaler saved")
