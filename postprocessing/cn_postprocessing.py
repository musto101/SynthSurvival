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
        # print('Inpute type before encoding:', x.dtype)
        # print('Input shape before encoding:', x.shape)
        if x.dtype == torch.int64:
            x = x.float()
        mu_logvar = self.encoder(x)
        # print('Output type after encoding:', mu_logvar.dtype)
        # print('Output shape after encoding:', mu_logvar.shape)
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

    # Rest of the code remains the same

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


mci = pd.read_csv('data/cn_preprocessed_wo_csf3.csv')

# change columns with less than 3 unique values to bool
for col in mci.columns:
    if mci[col].nunique() < 3:
        mci[col] = mci[col].astype(bool)


# mci.dtypes

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

# # Encode ordinal data
# le = LabelEncoder()
# X_ord = le.fit_transform(X_ord.reshape(-1))

# Convert to tensors
X_bin = torch.tensor(X_bin, dtype=torch.float32)
X_ord = torch.tensor(X_ord, dtype=torch.float32)
X_cont = torch.tensor(X_cont, dtype=torch.float32)

batch_size = 32
train_loader = [X_bin, X_ord, X_cont]
#
# # Define model
# input_dims = [X_bin.shape[1], X_ord.shape[1], X_cont.shape[1]]  # Binary, ordinal, continuous
# hidden_dims = [6, 6, 6]
# dropouts = [0.2, 0.2, 0.2]
# activations = [nn.Tanh, nn.Tanh, nn.Tanh]
# model = VAE(latent_dim=2, input_dims=input_dims, hidden_dims=hidden_dims, dropouts=dropouts, activations=activations)



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
#
# # plot losses
# import matplotlib.pyplot as plt
# plt.plot(losses)
# plt.show()

num_samples = 1000  # Number of synthetic samples to generate
z = torch.randn(num_samples, model.latent_dim)

with torch.no_grad():
    x_bin_synthetic = model.decoders[0](z)
    x_ord_synthetic = model.decoders[1](z)
    x_cont_synthetic = model.decoders[2](z)

# Binary data
binary_cols = mci.columns[mci.dtypes == bool]
x_bin_synthetic = torch.sigmoid(x_bin_synthetic) > 0.5

# check dimension of x_bin_synthetic and binary_cols
print(x_bin_synthetic.shape)
print(binary_cols.shape)

# Ordinal data preprocessing
x_ord_syn = torch.softmax(x_ord_synthetic, dim=1)
# convert to numpy
x_ord_synthetic = x_ord_syn.numpy()

# check dimension of x_ord_synthetic and ordinal_cols
print(x_ord_synthetic.shape)
print(ordinal_cols.shape)


# Continuous data
continuous_cols = mci.columns[mci.dtypes == float]
x_cont_synthetic = x_cont_synthetic

# check dimension of x_cont_synthetic and continuous_cols
print(x_cont_synthetic.shape)
print(continuous_cols.shape)

# convert to numpy
x_bin_synthetic = x_bin_synthetic.numpy()
# x_ord_synthetic = x_ord_synthetic.numpy()
x_cont_synthetic = x_cont_synthetic.numpy()

assert x_bin_synthetic.shape[1] == len(binary_cols), "Binary data dimension mismatch"
assert x_ord_synthetic.shape[1] == len(ordinal_cols), "Ordinal data dimension mismatch"
assert x_cont_synthetic.shape[1] == len(continuous_cols), "Continuous data dimension mismatch"

# change x_bin_synthetic to int
x_bin_synthetic = x_bin_synthetic.astype(int)

# create pandas dataframe
synthetic_data = np.concatenate((x_bin_synthetic, x_ord_synthetic, x_cont_synthetic), axis=1)
synthetic_data.shape
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
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
#
#
# class EncoderSubnet(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_dim, dropout, activation=nn.Tanh):
#         super(EncoderSubnet, self).__init__()
#         self.latent_dim = latent_dim
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation
#         output_dim = 2 * self.latent_dim
#
#         self.encoder = nn.Sequential(
#             self.dropout,
#             nn.Linear(self.input_dim, self.hidden_dim),
#             activation(),  # Apply activation directly
#             self.dropout,
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             activation(),
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             activation(),
#             nn.Linear(self.hidden_dim, output_dim)
#         )
#
#     def forward(self, x):
#         # print('Inpute type before encoding:', x.dtype)
#         # print('Input shape before encoding:', x.shape)
#         if x.dtype == torch.int64:
#             x = x.float()
#         mu_logvar = self.encoder(x)
#         # print('Output type after encoding:', mu_logvar.dtype)
#         # print('Output shape after encoding:', mu_logvar.shape)
#         mu = mu_logvar[:, : self.latent_dim]
#         logvar = mu_logvar[:, self.latent_dim :]
#         return mu, logvar
#
#
# class DecoderSubnet(nn.Module):
#     def __init__(self, latent_dim, output_dim, hidden_dim, dropout, activation=nn.Tanh):
#         super(DecoderSubnet, self).__init__()
#         self.latent_dim = latent_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation
#
#         self.decoder = nn.Sequential(
#             nn.Linear(self.latent_dim, self.hidden_dim),
#             activation(),
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             activation(),
#             nn.Linear(self.hidden_dim, self.output_dim)
#         )
#
#     def forward(self, z):
#         x_hat = self.decoder(z)
#         return x_hat
#
#
# class VAE(nn.Module):
#     def __init__(self, latent_dim, input_dims, hidden_dims, dropouts, activations):
#         super(VAE, self).__init__()
#         self.latent_dim = latent_dim
#         self.input_dims = input_dims
#         self.hidden_dims = hidden_dims
#         self.dropouts = dropouts
#         self.activations = activations
#
#         self.encoders = nn.ModuleList([EncoderSubnet(input_dim, latent_dim, hidden_dim, dropout, activation)
#                                        for input_dim, hidden_dim, dropout, activation in zip(input_dims, hidden_dims, dropouts, activations)])
#
#         self.decoders = nn.ModuleList([DecoderSubnet(latent_dim, input_dim, hidden_dim, dropout, activation)
#                                        for input_dim, hidden_dim, dropout, activation in zip(input_dims, hidden_dims, dropouts, activations)])
#
#     # Rest of the code remains the same
#
#     def reparameterize(self, mu, logvar):
#
#             # print('Mu:', mu)
#             # print('Logvar:', logvar)
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return mu + eps * std
#
#     def forward(self, x):
#         # Encode
#         mus, logvars = [], []
#         z = None
#         for encoder, x_subnet in zip(self.encoders, x):
#             mu, logvar = encoder(x_subnet)
#             mus.append(mu)
#             logvars.append(logvar)
#
#         # Reparameterize
#         if z is None:
#             z = self.reparameterize(mu, logvar)
#                         # print('Z type:', z.dtype)
#         else:
#             z = torch.cat([z, self.reparameterize(mu, logvar)], dim=1)
#                         # print('Z type:', z.dtype)
#
#         # Decode
#         x_hats = []
#         for decoder, input_dim in zip(self.decoders, self.input_dims):
#             x_hat = decoder(z)
#             x_hats.append(x_hat.view(-1, input_dim))
#
#         return x_hats, mus, logvars
#
#
# cn = pd.read_csv('data/cn_preprocessed_wo_csf3.csv')
#
# # change columns with less than 3 unique values to bool
# for col in cn.columns:
#     if cn[col].nunique() < 3:
#         cn[col] = cn[col].astype(bool)
#
# # Separate binary, ordinal, and continuous columns
# binary_cols = cn.columns[cn.dtypes == bool]
# ordinal_cols = cn.columns[cn.dtypes == int]
# continuous_cols = cn.columns[cn.dtypes == float]
#
# # Preprocess data
# cn_np = cn.values
# X_bin = cn_np[:, [cn.columns.get_loc(col) for col in binary_cols]]
# # convert values in x_bin to bool
# X_bin = X_bin.astype(bool)
#
# X_ord = cn_np[:, [cn.columns.get_loc(col) for col in ordinal_cols]]
# X_ord = X_ord.astype(np.int64)
#
# X_cont = cn_np[:, [cn.columns.get_loc(col) for col in continuous_cols]]
# X_cont = X_cont.astype(np.float32)
#
# # # Encode ordinal data
# # le = LabelEncoder()
# # X_ord = le.fit_transform(X_ord.reshape(-1))
#
# # Convert to tensors
# X_bin = torch.tensor(X_bin, dtype=torch.float32)
# X_ord = torch.tensor(X_ord, dtype=torch.float32)
# X_cont = torch.tensor(X_cont, dtype=torch.float32)
#
# batch_size = 32
# train_loader = [X_bin, X_ord, X_cont]
# #
# # # Define model
# # input_dims = [X_bin.shape[1], X_ord.shape[1], X_cont.shape[1]]  # Binary, ordinal, continuous
# # hidden_dims = [6, 6, 6]
# # dropouts = [0.2, 0.2, 0.2]
# # activations = [nn.Tanh, nn.Tanh, nn.Tanh]
# # model = VAE(latent_dim=2, input_dims=input_dims, hidden_dims=hidden_dims, dropouts=dropouts, activations=activations)
#
#
#
# # Define loss functions
# def binary_loss(x_hat, x):
#     return nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
#
# def ordinal_loss(x_hat, x):
#     return nn.functional.cross_entropy(x_hat, x, reduction='sum')
#
# def continuous_loss(x_hat, x):
#     return nn.functional.mse_loss(x_hat, x, reduction='sum')
#
# def kl_div_loss(mu, logvar):
#     return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
# def loss_function(x_hats, x, mus, logvars):
#     bin_loss = binary_loss(x_hats[0], x[0])
#     ord_loss = ordinal_loss(x_hats[1], x[1])
#     cont_loss = continuous_loss(x_hats[2], x[2])
#     kld = sum([kl_div_loss(mu, logvar) for mu, logvar in zip(mus, logvars)])
#     return bin_loss + ord_loss + cont_loss + kld
#
#
# # define parameters
# latent_dim = 32
# input_dims = [X_bin.shape[1], X_ord.shape[1], X_cont.shape[1]]  # Binary, ordinal, continuous
# hidden_dims = [36, 25, 40]
# dropouts = [0.12584205424284753, 0.45338312970056566, 0.09982021434300384]
# activations = [nn.Tanh, nn.Tanh, nn.Tanh]
# model = VAE(latent_dim=latent_dim, input_dims=input_dims, hidden_dims=hidden_dims, dropouts=dropouts,
#             activations=activations)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.09982021434300384, weight_decay=0.09121261683218564)
#
# # Train model
# epochs = 1000
#
# losses = []
# # Training loop
# for epoch in range(1000):
#     print(f'Epoch {epoch + 1}')
#     # Forward pass
#     x_hats, mus, logvars = model(train_loader)
#
#     # Compute loss
#     total_loss = loss_function(x_hats, train_loader, mus, logvars)
#
#     # Backward pass and optimization
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()
#
#     losses.append(total_loss.item())
#
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
#
# # Assuming model and other parts are defined here...
#
# num_samples = 1000  # Number of synthetic samples to generate
# z = torch.randn(num_samples, model.latent_dim)
#
# with torch.no_grad():
#     x_bin_synthetic = model.decoders[0](z)
#     x_ord_synthetic = model.decoders[1](z)
#     x_cont_synthetic = model.decoders[2](z)
#
# # Binary data processing
# binary_cols = cn.columns[cn.dtypes == bool]
# x_bin_synthetic = torch.sigmoid(x_bin_synthetic) > 0.5
#
# # turn x_ord_synthetic to a pandas dataframe
# x_ord_synthetic = x_ord_synthetic.numpy()
# x_ord_synthetic = np.round(x_ord_synthetic).astype(int)
# x_ord_synthetic = pd.DataFrame(x_ord_synthetic, columns=ordinal_cols)
#
#
# # Ordinal data processing
# ordinal_cols = cn.columns[cn.dtypes == int]
# ordinal_indices = [x_ord_synthetic.columns.get_loc(col) for col in ordinal_cols]  # Get indices of ordinal columns
# cn_indices = [cn.columns.get_loc(col) for col in ordinal_cols]  # Get indices of ordinal columns in original data
#
# # x_ord_synthetic = x_ord_synthetic.numpy()
# # x_ord_synthetic = np.round(x_ord_synthetic).astype(int)  # Ensure integers
#
# # Clip the values to ensure they fall within the valid range of the original data
# for idx, col in zip(ordinal_indices, cn_indices):
#     original_min = cn.iloc[:, col].min()
#     original_max = cn.iloc[:, col].max()
#     x_ord_synthetic.iloc[:, idx] = np.clip(x_ord_synthetic.iloc[:, idx], original_min, original_max)
#
# x_ord_synthetic['last_visit'].value_counts()
#
# # Continuous data processing
# continuous_cols = cn.columns[cn.dtypes == float]
# x_cont_synthetic = x_cont_synthetic.numpy()
#
# # Ensure dimensions match
# assert x_bin_synthetic.shape[1] == len(binary_cols), "Binary data dimension mismatch"
# assert x_ord_synthetic.shape[1] == len(ordinal_cols), "Ordinal data dimension mismatch"
# assert x_cont_synthetic.shape[1] == len(continuous_cols), "Continuous data dimension mismatch"
#
# # Convert to numpy
# x_bin_synthetic = x_bin_synthetic.numpy().astype(int)
#
# # Create pandas DataFrame
# synthetic_data = np.concatenate((x_bin_synthetic, x_ord_synthetic, x_cont_synthetic), axis=1)
# columns = binary_cols.append(ordinal_cols).append(continuous_cols)
# synthetic_data = pd.DataFrame(synthetic_data, columns=columns)
#
# synthetic_data['last_DX'].value_counts()
#
# # Save generated data
# synthetic_data.to_csv('data/generated_cn_data.csv', index=False)
#
#
# #
# # # plot losses
# # import matplotlib.pyplot as plt
# # plt.plot(losses)
# # plt.show()
# #
# # num_samples = 1000  # Number of synthetic samples to generate
# # z = torch.randn(num_samples, model.latent_dim)
# #
# # with torch.no_grad():
# #     x_bin_synthetic = model.decoders[0](z)
# #     x_ord_synthetic = model.decoders[1](z)
# #     x_cont_synthetic = model.decoders[2](z)
# #
# # # Binary data
# # binary_cols = cn.columns[cn.dtypes == bool]
# # x_bin_synthetic = torch.sigmoid(x_bin_synthetic) > 0.5
# #
# # # check dimension of x_bin_synthetic and binary_cols
# # print(x_bin_synthetic.shape)
# # print(binary_cols.shape)
# #
# # # Ordinal data
# # ordinal_cols = cn.columns[cn.dtypes == int]
# # # convert x_ord_synthetic to int
# # x_ord_synthetic = x_ord_synthetic.numpy()
# # x_ord_synthetic = x_ord_synthetic.astype(int)
# #
# # # summarize ordinal data
# # np.min(x_ord_synthetic)
# #
# # # check dimension of x_ord_synthetic and ordinal_cols
# # print(x_ord_synthetic.shape)
# # print(ordinal_cols.shape)
# #
# #
# # # Continuous data
# # continuous_cols = cn.columns[cn.dtypes == float]
# # x_cont_synthetic = x_cont_synthetic
# #
# # # check dimension of x_cont_synthetic and continuous_cols
# # print(x_cont_synthetic.shape)
# # print(continuous_cols.shape)
# #
# # # convert to numpy
# # x_bin_synthetic = x_bin_synthetic.numpy()
# # # x_ord_synthetic = x_ord_synthetic.numpy()
# # x_cont_synthetic = x_cont_synthetic.numpy()
# #
# # assert x_bin_synthetic.shape[1] == len(binary_cols), "Binary data dimension mismatch"
# # assert x_ord_synthetic.shape[1] == len(ordinal_cols), "Ordinal data dimension mismatch"
# # assert x_cont_synthetic.shape[1] == len(continuous_cols), "Continuous data dimension mismatch"
# #
# # # change x_bin_synthetic to int
# # x_bin_synthetic = x_bin_synthetic.astype(int)
# #
# # # create pandas dataframe
# # synthetic_data = np.concatenate((x_bin_synthetic, x_ord_synthetic, x_cont_synthetic), axis=1)
# # synthetic_data.shape
# # columns = binary_cols.append(ordinal_cols).append(continuous_cols)
# # synthetic_data = pd.DataFrame(synthetic_data, columns=columns)
# #
# # # save generated data
# # synthetic_data.to_csv('data/generated__cn_data.csv')