def register_iris_model(ws, model_dir="outputs", model_name="iris_model"):
    import os
    print(f"[INFO] Registering model folder '{model_name}' from directory '{model_dir}'...")

    print("[INFO] Files in model_dir:")
    for file in os.listdir(model_dir):
        print(" -", file)

    from azureml.core import Model
    model = Model.register(
        workspace=ws,
        model_path=model_dir,  # register entire folder (model.pkl + scaler.pkl)
        model_name=model_name,
        description="Iris classification model with scaler"
    )

    print(f"[INFO] Model registered: {model.name} (v{model.version})")
