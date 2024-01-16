import kfp
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath
from functools import partial


@partial(create_component_from_func, base_image="hbjs97/kfp:0.0.5")
def mnist_model_train(
    model_path: OutputPath("dill"),
    input_example_path: OutputPath("dill"),
    signature_path: OutputPath("dill"),
    conda_env_path: OutputPath("dill"),
):
    import dill
    from mlflow.models.signature import infer_signature, ModelSignature
    from mlflow.utils.environment import _mlflow_conda_env
    from mlflow.types.schema import Schema, TensorSpec
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, 10)
            self.fc2 = nn.Linear(10, 2)
            self.to(torch.float64)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif isinstance(x, pd.DataFrame):
                x = torch.tensor(x.values)

            x = x.view(-1, 4)
            if x.dtype != torch.float64:
                x = x.to(torch.float64)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
            x = x.detach().cpu().numpy()
            return x

    model = Net()

    model.eval()

    with open(model_path, "wb") as f:
        torch.save(model.state_dict(), f)

    input_example = torch.rand(1, 1, 2, 2)

    # Get the model's output as a PyTorch tensor
    output_numpy = model(input_example)

    # Use the numpy array for the signature
    signature = infer_signature(input_example.numpy(), output_numpy)

    with open(input_example_path, "wb") as f:
        dill.dump(input_example, f)

    with open(signature_path, "wb") as f:
        dill.dump(signature, f)

    conda_env = _mlflow_conda_env()
    with open(conda_env_path, "wb") as f:
        dill.dump(conda_env, f)


@partial(create_component_from_func, base_image="hbjs97/kfp:0.0.5")
def upload_mlflow_artifacts(
    model_name: str,
    model_path: InputPath("dill"),
    input_example_path: InputPath("dill"),
    signature_path: InputPath("dill"),
    conda_env_path: InputPath("dill"),
):
    import os
    import dill
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import mlflow
    from mlflow.pytorch import save_model, autolog, log_model
    from mlflow.tracking.client import MlflowClient
    import pandas as pd
    import numpy as np

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, 10)
            self.fc2 = nn.Linear(10, 2)
            self.to(torch.float64)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif isinstance(x, pd.DataFrame):
                x = torch.tensor(x.values)

            x = x.view(-1, 4)
            if x.dtype != torch.float64:
                x = x.to(torch.float64)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
            x = x.detach().cpu().numpy()
            return x

    with open(model_path, mode="rb") as f:
        model = Net()
        model.load_state_dict(torch.load(f))
        model.eval()

    with open(input_example_path, "rb") as f:
        input_example = dill.load(f)

    with open(signature_path, "rb") as f:
        signature = dill.load(f)

    # with open(conda_env_path, "rb") as f:
    #     conda_env = dill.load(f)

    if isinstance(input_example, torch.Tensor):
        input_example = input_example.numpy()
    elif isinstance(input_example, list):
        input_example = pd.DataFrame(input_example)

    save_model(
        pytorch_model=model,
        path=model_name,
        # conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )

    run = client.create_run("0")
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
