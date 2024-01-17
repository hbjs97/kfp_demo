import os
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


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


model_path = "./model.pth"

model = Net()
with open(model_path, "wb") as f:
    dill.dump(model, f)

with open(model_path, mode="rb") as f:
    model = dill.load(f)
    model.eval()
    a = model(torch.rand(1, 1, 2, 2))
    print(a)
