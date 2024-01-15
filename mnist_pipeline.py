import kfp
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath
from functools import partial


@partial(
    create_component_from_func,
    packages_to_install=["torch", "torchvision", "numpy", "dill"],
)
def mnist_model_train(
    model_path: OutputPath("dill"),
    input_example_path: OutputPath("dill"),
    signature_path: OutputPath("dill"),
    conda_env_path: OutputPath("dill"),
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import dill
    from torchvision import transforms
    from mlflow.models.signature import infer_signature
    from mlflow.utils.environment import _mlflow_conda_env

    # Define the neural network model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    # Create and save the model
    model = Net()

    # Prepare an input example and infer the signature
    input_example = torch.randn(1, 1, 28, 28)  # Example input tensor
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output_example = model(input_example)

    # Save the model
    with open(model_path, "wb") as f:
        dill.dump(model, f)

    # Save the input example
    with open(input_example_path, "wb") as f:
        dill.dump(input_example, f)

    # Infer and save the signature
    signature = infer_signature(input_example.numpy(), output_example.numpy())
    with open(signature_path, "wb") as f:
        dill.dump(signature, f)

    # Create and save the conda environment
    conda_env = _mlflow_conda_env(
        additional_pip_deps=["torch", "numpy", "dill", "mlflow"]
    )
    with open(conda_env_path, "wb") as f:
        dill.dump(conda_env, f)


@partial(
    create_component_from_func,
    packages_to_install=["torch", "numpy", "dill", "mlflow", "boto3"],
)
def upload_mlflow_artifacts(
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
    name="PyTorch MNIST Prediction Pipeline",
    description="Pipeline for predicting MNIST dataset using a PyTorch model.",
)
def pytorch_mnist_pipeline(model_name: str):
    model = mnist_model_train()
    _ = upload_mlflow_artifacts(
        model_name=model_name,
        model=model.outputs["model"],
        input_example=model.outputs["input_example"],
        signature=model.outputs["signature"],
        conda_env=model.outputs["conda_env"],
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pytorch_mnist_pipeline, "mnist_pipeline.yaml")
