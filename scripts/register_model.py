def register_model(ws, model_path, model_name="iris_model"):
    print(f"[INFO] Registering model {model_name}...")
    from azureml.core import Model
    model = Model.register(
        workspace=ws,
        model_path=model_path,
        model_name=model_name,
        description="Iris classification model"
    )
    print(f"[INFO] Model registered: {model.name} (v{model.version})")
