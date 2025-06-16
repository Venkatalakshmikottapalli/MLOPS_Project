def train_model(X_train, y_train):
    print("[INFO] Training model...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("[INFO] Model trained.")
    return model

def save_model(model, path="outputs/model.pkl"):
    import os
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved at {path}")
