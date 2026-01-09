import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Normalizes data between layers

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, 32)

        self.fc5 = nn.Linear(32, output_dim)

        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.leaky_relu(self.bn3(self.fc3(x)))

        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x
