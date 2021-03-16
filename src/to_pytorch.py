from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def to_pytorch(weights, biases, sizes, activation=nn.ReLU):
    model = SimpleClassifier(sizes, activation)
    state_dict = OrderedDict()
    for i, (a, b) in enumerate(zip(weights, biases)):
        weight_tensor = torch.tensor(a.transpose())
        bias_tensor = torch.tensor(b.transpose())
        state_dict.update({f"model.fc{i}.weight": weight_tensor})
        state_dict.update({f"model.fc{i}.bias": bias_tensor})

    model.load_state_dict(state_dict)
    model.double()
    model.eval()
    return model


class SimpleClassifier(nn.Module):
    def __init__(self, sizes, activation):
        super().__init__()
        layers = OrderedDict()
        for i in range(len(sizes) - 2):
            layers.update({f"fc{i}": nn.Linear(sizes[i], sizes[i + 1])})
            layers.update({f"activation{i}": activation()})

        layers.update({f"fc{len(sizes) - 2}": nn.Linear(sizes[len(sizes) - 2], sizes[len(sizes) - 1])})
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model.forward(x)


if __name__ == "__main__":
    seed = 42
    sizes = [10, 10, 10, 1]
    __cheat_A, __cheat_B = np.load("../models/" + str(seed) + "_" + "-".join(map(str, sizes)) + ".npy",
                                   allow_pickle=True)


    def matmul(a, b, c, np=np):
        if c is None:
            c = np.zeros(1)

        return np.dot(a, b) + c


    def run(x, inner_A, inner_B):
        for i, (a, b) in enumerate(zip(inner_A, inner_B)):
            # Compute the matrix product.
            # This is a right-matrix product which means that rows/columns are flipped
            # from the definitions in the paper.
            # This was the first method I wrote and it doesn't make sense.
            # Please forgive me.
            x = matmul(x, a, b)
            if i < len(sizes) - 2:
                x = x * (x > 0)
        return x


    model = to_pytorch(__cheat_A, __cheat_B, sizes)

    x = [-108, -56, -78, 75, 97, 10, -12, 85, 99, -145]
    y_old = run(x, __cheat_A, __cheat_B)
    y_new = model(torch.tensor(x).double())
    print(y_old)
    print(y_new)
    if float(y_old[0]) - float(y_new[0]) > 1e-4:
        print("NOT SAME")
