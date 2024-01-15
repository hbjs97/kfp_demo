import kfp
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath
from functools import partial


@partial(
    create_component_from_func,
    packages_to_install=["torch", "dill", "numpy"],
)
def load_data(
    x_path: OutputPath("dill"),
    y_path: OutputPath("dill"),
):
    import torch
    import dill

    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    with open(x_path, "wb") as f:
        dill.dump(x, f)
    with open(y_path, "wb") as f:
        dill.dump(y, f)


@partial(
    create_component_from_func,
    packages_to_install=["torch", "torchvision", "dill", "numpy", "mlflow"],
)
def train_model(
    x_path: InputPath("dill"),
    y_path: InputPath("dill"),
    model_path: OutputPath("dill"),
    input_example_path: OutputPath("dill"),
    signature_path: OutputPath("dill"),
    conda_env_path: OutputPath("dill"),
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import dill
    from mlflow.models.signature import infer_signature
    from mlflow.utils.environment import _mlflow_conda_env

    with open(x_path, "rb") as f:
        x = dill.load(f)
    with open(y_path, "rb") as f:
        y = dill.load(f)

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(2):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    with open(model_path, "wb") as f:
        dill.dump(model, f)

    input_example = x[0:1].cpu().numpy()  # Convert to numpy array
    with open(input_example_path, "wb") as f:
        dill.dump(input_example, f)

    x_numpy = x.cpu().numpy()  # Convert input x to numpy array
    model.eval()  # Set model to evaluation mode
    output_numpy = (
        model(torch.from_numpy(x_numpy)).detach().cpu().numpy()
    )  # Get model output as numpy array
    signature = infer_signature(
        x_numpy, output_numpy
    )  # Use numpy arrays for infer_signature
    with open(signature_path, "wb") as f:
        dill.dump(signature, f)

    conda_env = _mlflow_conda_env(
        additional_pip_deps=["dill", "pandas", "torch", "mlflow"]
    )
    with open(conda_env_path, "wb") as file_writer:
        dill.dump(conda_env, file_writer)


@partial(
    create_component_from_func,
    packages_to_install=["dill", "mlflow", "torch", "boto3"],
)
def upload_torch_model_to_mlflow(
    model_name: str,
    model_path: InputPath("dill"),
    input_example_path: InputPath("dill"),
    signature_path: InputPath("dill"),
    conda_env_path: InputPath("dill"),
):
    import os
    import dill
    from mlflow.pytorch import save_model
    from mlflow.tracking.client import MlflowClient

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    with open(model_path, mode="rb") as f:
        model = dill.load(f)

    with open(input_example_path, "rb") as f:
        input_example = dill.load(f)

    with open(signature_path, "rb") as f:
        signature = dill.load(f)

    with open(conda_env_path, "rb") as f:
        conda_env = dill.load(f)

    save_model(
        pytorch_model=model,
        path=model_name,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )
    run = client.create_run(experiment_id="0")
    client.log_artifact(run.info.run_id, model_name)


@dsl.pipeline(
    name="Simple PyTorch Training Pipeline",
    description="An example pipeline that trains a simple PyTorch model.",
)
def pytorch_training_pipeline(model_name: str):
    load_data_result = load_data()
    model = train_model(load_data_result.outputs["x"], load_data_result.outputs["y"])
    _ = upload_torch_model_to_mlflow(
        model_name=model_name,
        model=model.outputs["model"],
        input_example=model.outputs["input_example"],
        signature=model.outputs["signature"],
        conda_env=model.outputs["conda_env"],
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pytorch_training_pipeline, "pytorch_training_pipeline.yaml"
    )
