def evaluate_model(model, X_test, y_test, label_encoder):
    print("[INFO] Evaluating model...")
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("[INFO] Evaluation complete.")
    return acc, report
