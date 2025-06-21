from azureml.core import Workspace, Environment, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig
import os

# Load workspace from config.json
ws = Workspace.from_config()

# Load registered model
model = Model(ws, name="iris_model")

# Create environment from your environment.yml file
env = Environment.from_conda_specification(name="mlops-iris-env", file_path="environment.yml")

# Define inference configuration (points to score.py)
inference_config = InferenceConfig(
    entry_script="score.py",  # This is the scoring script
    environment=env
)

# Define deployment config (Azure Container Instance)
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    auth_enabled=True  # Enables authentication (can disable for testing)
)

# Deploy model as web service
service_name = "iris-mlservice"
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True  # Overwrite if service already exists
)

service.wait_for_deployment(show_output=True)

# Print scoring URI
print("Service state:", service.state)
print("Scoring URI:", service.scoring_uri)
print(service.get_logs())

