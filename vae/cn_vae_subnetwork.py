from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import optuna

class EncoderSubnet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout, activation=nn.Tanh):
        super(EncoderSubnet, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        output_dim = 2 * self.latent_dim

        self.encoder = nn.Sequential(
            self.dropout,
            nn.Linear(self.input_dim, self.hidden_dim),
            activation(),  # Apply activation directly
            self.dropout,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dtype == torch.int64:
            x = x.float()
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, : self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim :]
        return mu, logvar
class DecoderSubnet(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, dropout, activation=nn.Tanh):
        super(DecoderSubnet, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dims, hidden_dims, dropouts, activations):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.activations = activations

        self.encoders = nn.ModuleList([EncoderSubnet(input_dim, latent_dim, hidden_dim, dropout, activation)
                                       for input_dim, hidden_dim, dropout, activation in zip(input_dims, hidden_dims, dropouts, activations)])

        self.decoders = nn.ModuleList([DecoderSubnet(latent_dim, input_dim, hidden_dim, dropout, activation)
                                       for input_dim, hidden_dim, dropout, activation in zip(input_dims, hidden_dims, dropouts, activations)])

    def reparameterize(self, mu, logvar):

            # print('Mu:', mu)
            # print('Logvar:', logvar)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def forward(self, x):
        # Encode
        mus, logvars = [], []
        z = None
        for encoder, x_subnet in zip(self.encoders, x):
            mu, logvar = encoder(x_subnet)
            mus.append(mu)
            logvars.append(logvar)

        # Reparameterize
        if z is None:
            z = self.reparameterize(mu, logvar)
                        # print('Z type:', z.dtype)
        else:
            z = torch.cat([z, self.reparameterize(mu, logvar)], dim=1)
                        # print('Z type:', z.dtype)

        # Decode
        x_hats = []
        for decoder, input_dim in zip(self.decoders, self.input_dims):
            x_hat = decoder(z)
            x_hats.append(x_hat.view(-1, input_dim))

        return x_hats, mus, logvars

cn = pd.read_csv('data/cn_preprocessed_wo_csf3.csv')

# change columns with less than 3 unique values to bool
for col in cn.columns:
    if cn[col].nunique() < 3:
        cn[col] = cn[col].astype(bool)

# Separate binary, ordinal, and continuous columns
binary_cols = cn.columns[cn.dtypes == bool]
ordinal_cols = cn.columns[cn.dtypes == int]
continuous_cols = cn.columns[cn.dtypes == float]

# Preprocess data
cn_np = cn.values
X_bin = cn_np[:, [cn.columns.get_loc(col) for col in binary_cols]]
# convert values in x_bin to bool
X_bin = X_bin.astype(bool)

X_ord = cn_np[:, [cn.columns.get_loc(col) for col in ordinal_cols]]
X_ord = X_ord.astype(np.float32)

X_cont = cn_np[:, [cn.columns.get_loc(col) for col in continuous_cols]]
X_cont = X_cont.astype(np.float32)


# Convert to tensors
X_bin = torch.tensor(X_bin, dtype=torch.float32)
X_ord = torch.tensor(X_ord, dtype=torch.float32)
X_cont = torch.tensor(X_cont, dtype=torch.float32)

batch_size = 32
train_loader = [X_bin, X_ord, X_cont]

# Define loss functions
def binary_loss(x_hat, x):
    return nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='mean')

def ordinal_loss(x_hat, x):
    return nn.functional.cross_entropy(x_hat, x, reduction='mean')

def continuous_loss(x_hat, x):
    return nn.functional.mse_loss(x_hat, x, reduction='mean')

def kl_div_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_function(x_hats, x, mus, logvars):
    bin_loss = binary_loss(x_hats[0], x[0])
    ord_loss = ordinal_loss(x_hats[1], x[1])
    cont_loss = continuous_loss(x_hats[2], x[2])
    kld = sum([kl_div_loss(mu, logvar) for mu, logvar in zip(mus, logvars)])
    return bin_loss + ord_loss + cont_loss + kld

# create bayesian optimization function

def objective(trial):
    # Define model
    input_dims = [X_bin.shape[1], X_ord.shape[1], X_cont.shape[1]]  # Binary, ordinal, continuous
    latent_dims = trial.suggest_int('latent_dim', 2, 36)
    hidden_dims = [trial.suggest_int('hidden_dim1', 10, 40),
                   trial.suggest_int('hidden_dim2', 10, 40),
                   trial.suggest_int('hidden_dim3', 10, 40)]
    dropouts = [trial.suggest_float('dropout1', 0.1, 0.5),
                trial.suggest_float('dropout2', 0.1, 0.5),
                trial.suggest_float('dropout3', 0.1, 0.5)]
    activations = [nn.Tanh, nn.Tanh, nn.Tanh]
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-1)
    model = VAE(latent_dim=latent_dims, input_dims=input_dims, hidden_dims=hidden_dims, dropouts=dropouts, activations=activations)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    # Training loop
    for epoch in range(1000):
        # Forward pass
        x_hats, mus, logvars = model(train_loader)

        # Compute loss
        total_loss = loss_function(x_hats, train_loader, mus, logvars)

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

    return total_loss.item()


# Perform optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000, n_jobs=-1)

# Get best parameters
best_params = study.best_params

# Print best parameters
print("Best parameters: ", best_params)

# Get best loss
best_loss = study.best_value

# Print best loss
print("Best loss: ", best_loss)

# run the model with the best parameters
# Define model
input_dims = [X_bin.shape[1], X_ord.shape[1], X_cont.shape[1]]  # Binary, ordinal, continuous
hidden_dims = [best_params['hidden_dim1'], best_params['hidden_dim2'], best_params['hidden_dim3']]
dropouts = [best_params['dropout1'], best_params['dropout2'], best_params['dropout3']]
activations = [nn.Tanh, nn.Tanh, nn.Tanh]
lr = best_params['lr']
weight_decay = best_params['weight_decay']
model = VAE(latent_dim=2, input_dims=input_dims, hidden_dims=hidden_dims, dropouts=dropouts, activations=activations)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

losses = []
# Training loop
for epoch in range(1000):
    # Forward pass
    x_hats, mus, logvars = model(train_loader)

    # Compute loss
    total_loss = loss_function(x_hats, train_loader, mus, logvars)

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    losses.append(total_loss.item())

# save losses
np.save('losses.npy', losses)

# # save model
torch.save(model.state_dict(), 'cn_model.pth')
