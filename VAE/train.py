import torch
from torch import optim
from ResVAE import ResVariationalAutoEncoder  # Assuming you have a VAE model defined in 'model.py'
from torch.utils.data import TensorDataset, DataLoader
import h5py
import numpy as np

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 100  # Adjusted input dimension to match your complex data
H_DIM = 32
H_LAYERS = [2,2,2]
Z_DIM = 16
NUM_EPOCHS = 5000
BATCH_SIZE = 1024 # Adjusted batch size
LR_RATE = 1e-3

# Load the dataset to a 777x4x1000 data called conformalWeldings
# Load the .mat file
#with h5py.File('./VAE/preprocessed2.mat', 'r') as mat_file:

#with h5py.File('./VAE/preprocessed_normalized.mat', 'r') as mat_file:
MAT_PATH = '../data/preprocessed.mat'

def load_cw(mat_path):
    with h5py.File(mat_path, 'r') as mat_file:
        # Access the 'bc_dict' group
        bc_dict_group = mat_file['bc_dict']
        # Initialize a 3D NumPy array to store theta
        conformalWeldings = np.empty((len(bc_dict_group), 100), dtype=np.float32)  # Use float32 instead of complex128

        # Iterate over the fields (e.g., 'Case00_12', 'Case00_13', etc.)
        for i, field_name in enumerate(bc_dict_group):
            case_group = bc_dict_group[field_name]
            # xq = case_group['x'][:]  # Load the 'x' dataset into a structured array
            # yq = case_group['y'][:]  # Load the 'y' dataset into a structured array
            theta = case_group['theta'][:]  # Load the 'theta' dataset into a structured array
            theta = np.insert(theta, 100, 2*np.pi)
            theta = np.diff(theta) # Use diff between theta to train
            # theta_ma = case_group['theta_ma'][:]  # Load the 'theta_ma' dataset into a structured array
            # theta_ma = np.insert(theta_ma, 0, 0)
            # theta_ma = np.diff(theta_ma)
            # bias = case_group['bias'][:]  # Load the 'bias' dataset into a structured array
            conformalWeldings[i] = np.log(1/theta) # Correspond theta and bias together
    return conformalWeldings

# Assuming you have your 777x2x1000 data stored in a NumPy array named 'conformalWeldings'
#conformalWeldings = conformalWeldings[:10, :, :]

def main():
    cw = load_cw(MAT_PATH)
    cw_tensor = torch.tensor(cw).to(DEVICE)

    train_data = TensorDataset(cw_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    
    model = ResVariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=H_DIM, h_layers=H_LAYERS, z_dim=Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=1e-5, betas=(0.9, 0.999))

    model.to(DEVICE)
    kl_rate = 0.01
    
    # Start Training
    model.train()
    
    loader_size = len(train_loader)
    loss_list = np.zeros(loader_size)  # To store reconstruction diff losses
    recon_loss_list = np.zeros(loader_size)  # To store reconstruction losses
    kl_loss_list = np.zeros(loader_size)         # To store KL divergence losses

    for epoch in range(NUM_EPOCHS):
        for i, [cw] in enumerate(train_loader):
            cw = cw.to(DEVICE, dtype=torch.float32).view(cw.shape[0], INPUT_DIM)
            x_reconstructed, mu, log_var = model(cw)

            # Compute loss
            loss, recon_loss, kl_loss = model.loss(x_reconstructed, cw, mu, log_var, kl_rate)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Append losses to the lists
            loss_list[i] = loss.item()
            recon_loss_list[i] = recon_loss.item()
            kl_loss_list[i] = kl_loss.item()

        # Calculate and print average losses for this epoch
        avg_loss = loss_list.mean()
        avg_recon_loss = recon_loss_list.mean()
        avg_kl_loss = kl_loss_list.mean()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}, Reconstruction: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
            
        if epoch % 100 == 0:
            torch.save(model, './VAE_theta+bias1.pth')
