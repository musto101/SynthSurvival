import pandas as pd
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata


def match_distributions_fixed(source, target):
    """
    Adjust the distribution of source to match the target using ECDF matching.

    Parameters:
    source (pd.Series): The source data to be adjusted.
    target (pd.Series): The target data to match the distribution to.

    Returns:
    pd.Series: The adjusted source data.
    """
    # Rank the source data
    source_ranks = rankdata(source, method='ordinal')
    # Sort the target data
    target_sorted = np.sort(target)

    # Ensure target_sorted has the same length as source_ranks
    if len(target_sorted) < len(source_ranks):
        target_sorted = np.concatenate(
            [target_sorted, np.repeat(target_sorted[-1], len(source_ranks) - len(target_sorted))])

    # Map source ranks to target sorted values
    target_mapped = target_sorted[source_ranks - 1]

    return target_mapped


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

        else:
            z = torch.cat([z, self.reparameterize(mu, logvar)], dim=1)

        # Decode
        x_hats = []
        for decoder, input_dim in zip(self.decoders, self.input_dims):
            x_hat = decoder(z)
            x_hats.append(x_hat.view(-1, input_dim))

        return x_hats, mus, logvars


mci = pd.read_csv('data/cn_preprocessed_wo_csf3.csv')

# change columns with less than 3 unique values to bool
for col in mci.columns:
    if mci[col].nunique() < 3:
        mci[col] = mci[col].astype(bool)


# Separate binary, ordinal, and continuous columns
binary_cols = mci.columns[mci.dtypes == bool]
ordinal_cols = mci.columns[mci.dtypes == int]
continuous_cols = mci.columns[mci.dtypes == float]

# Preprocess data
mci_np = mci.values
X_bin = mci_np[:, [mci.columns.get_loc(col) for col in binary_cols]]
# convert values in x_bin to bool
X_bin = X_bin.astype(bool)

X_ord = mci_np[:, [mci.columns.get_loc(col) for col in ordinal_cols]]
X_ord = X_ord.astype(np.int64)

X_cont = mci_np[:, [mci.columns.get_loc(col) for col in continuous_cols]]
X_cont = X_cont.astype(np.float32)

# Convert to tensors
X_bin = torch.tensor(X_bin, dtype=torch.float32)
X_ord = torch.tensor(X_ord, dtype=torch.float32)
X_cont = torch.tensor(X_cont, dtype=torch.float32)

batch_size = 32
train_loader = [X_bin, X_ord, X_cont]

# Define loss functions
def binary_loss(x_hat, x):
    return nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')

def ordinal_loss(x_hat, x):
    return nn.functional.cross_entropy(x_hat, x, reduction='sum')

def continuous_loss(x_hat, x):
    return nn.functional.mse_loss(x_hat, x, reduction='sum')

def kl_div_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_function(x_hats, x, mus, logvars):
    bin_loss = binary_loss(x_hats[0], x[0])
    ord_loss = ordinal_loss(x_hats[1], x[1])
    cont_loss = continuous_loss(x_hats[2], x[2])
    kld = sum([kl_div_loss(mu, logvar) for mu, logvar in zip(mus, logvars)])
    return bin_loss + ord_loss + cont_loss + kld


# define parameters
latent_dim = 15
input_dims = [X_bin.shape[1], X_ord.shape[1], X_cont.shape[1]]  # Binary, ordinal, continuous
hidden_dims = [26, 32, 40]
dropouts = [0.4878913724469232, 0.284036600347419, 0.364191346646578]
activations = [nn.Tanh, nn.Tanh, nn.Tanh]
model = VAE(latent_dim=latent_dim, input_dims=input_dims, hidden_dims=hidden_dims, dropouts=dropouts,
            activations=activations)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.09972751965351388, weight_decay=0.036068718215822596)

# Train model
epochs = 1000

losses = []
# Training loop
for epoch in range(1000):
    print(f'Epoch {epoch + 1}')
    # Forward pass
    x_hats, mus, logvars = model(train_loader)

    # Compute loss
    total_loss = loss_function(x_hats, train_loader, mus, logvars)

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    losses.append(total_loss.item())

num_samples = 1000  # Number of synthetic samples to generate
z = torch.randn(num_samples, model.latent_dim)

with torch.no_grad():
    x_bin_synthetic = model.decoders[0](z)
    x_ord_synthetic = model.decoders[1](z)
    x_cont_synthetic = model.decoders[2](z)

# Binary data
binary_cols = mci.columns[mci.dtypes == bool]
x_bin_synthetic = torch.sigmoid(x_bin_synthetic) > 0.5

# Ordinal data preprocessing
x_ord_syn = torch.softmax(x_ord_synthetic, dim=1)
# convert to numpy
x_ord_synthetic = x_ord_syn.numpy()

# Continuous data
continuous_cols = mci.columns[mci.dtypes == float]
x_cont_synthetic = x_cont_synthetic

# convert to numpy
x_bin_synthetic = x_bin_synthetic.numpy()
# x_ord_synthetic = x_ord_synthetic.numpy()
x_cont_synthetic = x_cont_synthetic.numpy()

# change x_bin_synthetic to int
x_bin_synthetic = x_bin_synthetic.astype(int)

# create pandas dataframe
synthetic_data = np.concatenate((x_bin_synthetic, x_ord_synthetic, x_cont_synthetic), axis=1)

columns = binary_cols.append(ordinal_cols).append(continuous_cols)
synthetic_data = pd.DataFrame(synthetic_data, columns=columns)

synthetic_data['last_visit'].describe()

# Extract the last_visit columns
mci_last_visit = mci['last_visit']
synth_last_visit = synthetic_data['last_visit']

# Match the distribution
generated_last_visit_matched_fixed = match_distributions_fixed(synth_last_visit, mci_last_visit)

# Replace the 'last_visit' column in the generated dataframe with the matched values
synthetic_data['last_visit'] = generated_last_visit_matched_fixed

# check the distribution of the last_visit column
synthetic_data['last_visit'].describe()

# save generated data
synthetic_data.to_csv('data/generated_cn_data.csv')
