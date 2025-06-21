from azureml.core import Workspace

# Connect to your workspace (same way you did in deploy.py)
ws = Workspace.from_config()

# Get the deployed service by name
service = ws.webservices['iris-mlservice']  # replace with your actual service name

# Get the primary key (first key)
primary_key = service.get_keys()[0]

print("Primary key for authentication:", primary_key)
