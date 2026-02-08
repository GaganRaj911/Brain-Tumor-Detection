import numpy as np
import cv2
import joblib
from Features_codes.HOG import extract_hog_features

scaler=joblib.load('Model/feature_scaler.pkl')
model=joblib.load('Model/brain_tumor_model.pkl')

def predict_from_image(img):
    image_size=(128, 128)
    img = cv2.resize(img, image_size)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img / 255.0  # Normalize to [0, 1]

    data=np.array([img])

    hog=extract_hog_features(data)

    features=hog

    features_scaled=scaler.transform(features)

    prediction=model.predict(features_scaled)

    label = "Tumor Detected" if prediction[0] == 1 else "No Tumor"
    return label
