from azureml.core import Workspace, Dataset

def get_workspace():
    """Connect to Azure ML Workspace"""
    ws = Workspace.from_config()
    return ws

def load_data(ws, dataset_name="iris"):
    """Load dataset from Azure"""
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    df = dataset.to_pandas_dataframe()
    return df

if __name__ == "__main__":
    ws = get_workspace()
    df = load_data(ws, dataset_name="iris")
    print(df.head())
