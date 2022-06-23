import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Optimization():
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.train_losses = []
        self.val_losses = []

    def train_step_l1(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)

        L1_reg = torch.tensor(0., requires_grad=True, device=self.device)
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        # l1_norm = torch.norm(self.model.parameters(), p=1)
        loss = loss + 1e-3 * L1_reg
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train_step(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self, train_loader, val_loader, n_epochs):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # loss = self.train_step(x_batch, y_batch)
                loss = self.train_step_l1(x_batch, y_batch)
                batch_losses.append(loss)

            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    self.model.eval()
                    y_hat = self.model(x_batch)
                    loss = self.loss_fn(y_batch, y_hat).item()

                    L1_reg = torch.tensor(0., requires_grad=True)
                    for name, param in self.model.named_parameters():
                        if 'weight' in name:
                            L1_reg = L1_reg + torch.norm(param, 1)

                    # l1_norm = torch.norm(self.model.parameters(), p=1)
                    loss = loss + 1e-3 * L1_reg

                    batch_val_losses.append(loss.cpu())

                val_loss = np.mean(batch_val_losses)
                self.val_losses.append(val_loss)

            if (epoch < 10) | (epoch % 50 == 0):
                print(f'epoch{epoch}: train_loss:{train_loss:4f}\t val_loss:{val_loss:4f}')




class RobustNN(nn.Module):

    def __init__(self, input_dim, layer_dim, hidden_dim, output_dim, dropout_prob, device):
        super(RobustNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.device = device

        self.relu = nn.LeakyReLU(1e-2)
        self.dropout = nn.Dropout(dropout_prob)

        self.l1 = nn.Linear(input_dim, hidden_dim)

        self.h1 = nn.Linear(hidden_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # self.l1 = nn.Linear(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
        # self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.h1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.h2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.reshape(-1, 1)).double().to(self.device)
        y_hat = self.forward(X_t).detach().cpu().numpy().flatten()
        return y_hat

    def __name__(self):
        return 'RobustNN'

class NNLayer3(nn.Module):

    def __init__(self, input_dim, layer_dim, hidden_dim, output_dim, dropout_prob, device):
        super(NNLayer3, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.device = device

        self.relu = nn.LeakyReLU(1e-2)
        self.dropout = nn.Dropout(dropout_prob)
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.reshape(-1, 1)).double().to(self.device)
        y_hat = self.forward(X_t).detach().cpu().numpy().flatten()
        return y_hat

    def __name__(self):
        return 'NNLayer2'

class NNLayer2(nn.Module):

    def __init__(self, input_dim, layer_dim, hidden_dim, output_dim, dropout_prob, device):
        super(NNLayer2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.device = device

        self.relu = nn.LeakyReLU(1e-2)
        self.dropout = nn.Dropout(dropout_prob)
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.reshape(-1, 1)).double().to(self.device)
        y_hat = self.forward(X_t).detach().cpu().numpy().flatten()
        return y_hat

    def __name__(self):
        return 'NNLayer2'

class NNLayer1(nn.Module):

    def __init__(self, input_dim, layer_dim, hidden_dim, output_dim, dropout_prob, device):
        super(NNLayer1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.device = device

        self.relu = nn.LeakyReLU(1e-2)
        self.dropout = nn.Dropout(dropout_prob)

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.reshape(-1, 1)).double().to(self.device)
        y_hat = self.forward(X_t).detach().cpu().numpy().flatten()
        return y_hat

    def __name__(self):
        return 'NNLayer1'

class NNLayer1Sigmoid(nn.Module):

    def __init__(self, input_dim, layer_dim, hidden_dim, output_dim, dropout_prob, device):
        super(NNLayer1Sigmoid, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.device = device

        self.relu = nn.LeakyReLU(1e-2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.dropout(out)
        out = self.sigmoid(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.reshape(-1, 1)).double().to(self.device)
        y_hat = self.forward(X_t).detach().cpu().numpy().flatten()
        return y_hat

    def __name__(self):
        return 'NNLayer1Sigmoid'
