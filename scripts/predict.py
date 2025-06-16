import joblib
import numpy as np

def preprocess_sample(sample):
    """
    Preprocess raw sample to match training features:
    - sample: 2D numpy array with columns:
      sepal_length, sepal_width, petal_length, petal_width
    - Returns: processed 2D array with 6 features
    """
    sepal_length = sample[:, 0]
    sepal_width = sample[:, 1]
    petal_length = sample[:, 2]
    petal_width = sample[:, 3]

    sepal_ratio = sepal_length / sepal_width
    petal_ratio = petal_length / petal_width

    processed = np.column_stack((sepal_length, sepal_width, petal_length, petal_width, sepal_ratio, petal_ratio))
    return processed

def load_model(model_path="outputs/model.pkl", scaler_path="outputs/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

import pandas as pd

def predict(model, scaler, raw_sample):
    """
    raw_sample: 2D array or list with 4 features (no ratios)
    Returns predicted class label(s)
    """
    sample_processed = preprocess_sample(np.array(raw_sample))

    # Add column names matching training features
    feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'sepal_ratio', 'petal_ratio']
    sample_df = pd.DataFrame(sample_processed, columns=feature_columns)

    sample_scaled = scaler.transform(sample_df)
    preds = model.predict(sample_scaled)
    return preds


if __name__ == "__main__":
    model, scaler = load_model()

    # Example input: 1 sample with 4 raw features
    raw_sample = [[5.1, 3.5, 1.4, 0.2]]

    prediction = predict(model, scaler, raw_sample)
    print(f"Predicted class index: {prediction[0]}")
