import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

# Simple model for demonstration
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

def get_data():
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return X, y

def train(model, X, y):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return model.state_dict()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config): 
        return [val.cpu().numpy() for val in model.state_dict().values()]
    def fit(self, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        X, y = get_data()
        model_state = train(model, X, y)
        return [val.cpu().numpy() for val in model_state.values()], len(X), {}
    def evaluate(self, parameters, config):
        return 0.0, len(get_data()[0]), {}

model = Net()
fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())