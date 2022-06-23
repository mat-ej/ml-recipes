import numpy as np
import pandas as pd
from torch import optim
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torch.utils.data import DataLoader
from reg.torch_nn_regression import *
from reg.plots import *
from sklearn.preprocessing import StandardScaler
torch.manual_seed(0)
np.random.seed(0)

def get_model(model, params):
    models = {
        'nnlayer1':NNLayer1,
        'nnlayer1sigmoid':NNLayer1Sigmoid,
        'nnlayer2': NNLayer2,
        'nnlayer3': NNLayer3,
    }
    return models.get(model.lower())(**params)

if __name__ == "__main__":
    N=1000
    end = 1
    noise_end = .1
    X = np.linspace(0, end, N)
    X = X + np.random.normal(scale=0.2, size=[X.shape[0]])
    X[X<0] = 0
    f = lambda x: np.sqrt(x) - x
    noise = np.random.normal(0, noise_end, N)
    y = f(X) + noise
    # y[y<0] = 0



    val_noise = np.random.normal(0, noise_end, int(N / 10))
    # X_val = np.random.uniform(low=0, high=end, size=int(N / 10))
    X_val = np.linspace(0, end, int(N / 10))
    y_val = f(X_val) + val_noise

    # y_val[y_val<0] = 0

    Xt, yt = X.reshape(-1,1), y.reshape(-1,1)
    Xt_val, yt_val = X_val.reshape(-1,1), y_val.reshape(-1,1)

    scaler = StandardScaler()
    Xt = scaler.fit_transform(Xt)
    Xt_val = scaler.transform(Xt_val)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = SimpleDataset(Xt, yt)
    train_loader = DataLoader(train, batch_size=int(len(X)/30))

    val = SimpleDataset(Xt_val, yt_val)
    val_loader = DataLoader(val, batch_size=len(val))

    model_params = {
        'input_dim': 1,
        'output_dim': 1,
        'layer_dim': None,
        'hidden_dim': 1024,
        'dropout_prob': 0.2,
        'device': device
    }
    # train = Dataset(x, y)
    # train_loader = TensorDataset(x, y)


    model = get_model('NNLayer2', model_params)
    # model = get_model('NNLayer1', model_params)
    model.double()

    x0, y0 = next(iter(train_loader))
    loss_fn = MSELoss(reduction='mean')
    # loss_fn = SmoothL1Loss(reduction='mean')
    # loss_fn = L1Loss(reduction='mean')

    y0_hat = model(x0)
    loss = loss_fn(y0, y0_hat).detach()
    print(f"epoch0: train_loss: {loss:4f}")

    learning_rate = 1e-3
    weight_decay = 1e-5
    n_epochs = 300
    optimizer_params = {
        'lr' : learning_rate,
        'weight_decay' : weight_decay
    }
    optimizer = optim.Adam(model.parameters(), **optimizer_params)

    opt = Optimization(model, loss_fn, optimizer, device)

    opt.train(train_loader, val_loader, n_epochs)

    X_torch = torch.from_numpy(Xt).double().to(device)
    y_hat = model(X_torch).detach().cpu().numpy()

    plot_model_preds(X, y, f, model, Xt)


    plot_residuals_model(model, Xt, y)
    plot_scatter_pred_actual_model(model, Xt, y)

    # opt.train(train_loader, )


