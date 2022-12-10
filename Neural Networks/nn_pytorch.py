import torch.nn as nn
import torch

class NeuralNetworkPyTorch:
    def __init__(self, input_size, no_of_layers, act_func, hidden_size, lr = 0.001):
        self.lr = lr
        self.model = Model(input_size, no_of_layers, act_func, hidden_size)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X, Y, T):
        for t in range(T):
            self.optimizer.zero_grad()
            Y_pred = self.model(X)
            Y = Y.view(-1, 1)
            loss = self.loss_fn(Y_pred, Y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            Y_pred = self.model(X)
            Y_pred = (Y_pred > 0.5).float()
            return Y_pred

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        return torch.mean((Y_pred != Y).float())

    def get_weights(self):
        return self.model.state_dict()

class Model(nn.Module):
    def __init__(self, input_size, no_of_layers, activation_function, hidden_size):
        super(Model, self).__init__()
        self.no_of_layers = no_of_layers
        for i in range(no_of_layers):
            if i == 0:
                setattr(self, f"fc{i+1}", nn.Linear(input_size, hidden_size))
            else:
                setattr(self, f"fc{i+1}", nn.Linear(hidden_size, hidden_size))

        setattr(self, f"fc{no_of_layers+1}", nn.Linear(hidden_size, 1))
        
        if activation_function == "relu":
            self.act = nn.ReLU()
            for i in range(1, len(self._modules)):
                nn.init.kaiming_uniform_(getattr(self, f"fc{i}").weight)
        elif activation_function == "tanh":
            self.act = nn.Tanh()
            for i in range(1, len(self._modules)):
                nn.init.xavier_uniform_(getattr(self, f"fc{i}").weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(1, self.no_of_layers + 1):
            x = getattr(self, f"fc{i}")(x)
            x = self.act(x)
        
        x = getattr(self, f"fc{self.no_of_layers + 1}")(x)
        
        x = self.sigmoid(x)
        
        return x