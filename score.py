import os
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    global scaler

    # Get the directory where model is registered
    model_dir = Model.get_model_path("iris_model")

    # Load model and scaler from the registered folder
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))


def preprocess_sample(sample):
    sepal_length = sample[:, 0]
    sepal_width = sample[:, 1]
    petal_length = sample[:, 2]
    petal_width = sample[:, 3]

    sepal_ratio = sepal_length / sepal_width
    petal_ratio = petal_length / petal_width

    processed = np.column_stack((
        sepal_length, sepal_width, petal_length, petal_width,
        sepal_ratio, petal_ratio
    ))
    return processed


def run(raw_data):
    try:
        # Handle both string input and dict input
        if isinstance(raw_data, str):
            data = json.loads(raw_data)["data"]
        else:
            data = raw_data["data"]

        sample = np.array(data)
        sample_processed = preprocess_sample(sample)
        sample_scaled = scaler.transform(sample_processed)
        preds = model.predict(sample_scaled)

        return {"prediction": preds.tolist()}

    except Exception as e:
        return {"error": str(e)}
