import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class RegressionDataLoader(Dataset):
    def __init__(self, filepath):
        self.features, self.labels = self._load_data(filepath)
    
    def _load_data(self, filepath):
        data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        return data[:, :-1], data[:, -1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def build_neural_network(input_size, hidden_layer_size, output_size, depth, activation_function, device):
    layers = [nn.Linear(input_size, hidden_layer_size), activation_function()]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_layer_size, hidden_layer_size), activation_function()])
    layers.append(nn.Linear(hidden_layer_size, output_size))
    return nn.Sequential(*layers).to(device)

def apply_weight_initialization(network, initialization_function):
    for layer in network:
        if isinstance(layer, nn.Linear):
            initialization_function(layer.weight)

def perform_training(data_loader, model, criterion, optimizer):
    model.train()
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
    return loss.item()

def perform_testing(data_loader, model, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += criterion(predictions, targets).item()
    return total_loss / len(data_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

train_dataset = RegressionDataLoader('train.csv')
test_dataset = RegressionDataLoader('test.csv')
train_loader = DataLoader(train_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=20)

configurations = [(nn.ReLU, nn.init.kaiming_normal_), (nn.Tanh, nn.init.xavier_normal_)]
network_results = {}

for activation, initializer in configurations:
    for width in [5, 10, 25, 50, 100]:
        for depth in [3, 5, 9]:
            model = build_neural_network(4, width, 1, depth, activation, device)
            apply_weight_initialization(model, initializer)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            for epoch in range(15):
                train_loss = perform_training(train_loader, model, criterion, optimizer)
                test_loss = perform_testing(test_loader, model, criterion)

            print(f"Activation: {activation.__name__}, Width: {width}, Depth: {depth}, Train Loss: {train_loss}, Test Loss: {test_loss}")
            network_results[(activation.__name__, width, depth)] = (train_loss, test_loss)