from scripts.load_data import get_workspace, load_data
from scripts.preprocess import preprocess_data
from scripts.train import train_model, save_model
from scripts.evaluate import evaluate_model
from scripts.register_model import register_model


if __name__ == "__main__":
    # Step 1: Load data
    ws = get_workspace()
    df = load_data(ws)

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(df)

    # Step 3: Train
    model = train_model(X_train, y_train)
    save_model(model)

    # Step 4: Evaluate
    acc, report = evaluate_model(model, X_test, y_test, label_encoder)
    print(f"Accuracy: {acc:.4f}\n")
    print(report)

    # Step 5: Register
    register_model(ws, "outputs/model.pkl")
