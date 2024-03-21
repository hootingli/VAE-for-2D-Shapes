import torch
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder  # Assuming you have a VAE model defined in 'model.py'
from torch.utils.data import TensorDataset, DataLoader
import h5py
import pandas as pd
import numpy as np
import scipy.io
from torch.optim import lr_scheduler
from deeper_model import DeepVariationalAutoEncoder

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    NEW_LR_RATE = LR_RATE * (0.75 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = NEW_LR_RATE


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
INPUT_DIM = 200  # Adjusted input dimension to match your complex data
H_DIM = 200
Z_DIM = 2
NUM_EPOCHS = 1000
BATCH_SIZE = 16 # Adjusted batch size
LR_RATE = 5e-3

# Assuming you have a PyTorch model called VariationalAutoEncoder defined in 'model.py'
model = DeepVariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#optimizer = optim.Adagrad(model.parameters(), lr=LR_RATE)
#loss_fn = nn.BCELoss(reduction="sum")  # You might want to use a different loss function
#loss_fn = nn.L1Loss(reduction="sum")
loss_fn = nn.MSELoss(reduction="sum")


# Load the dataset to a 777x4x1000 data called conformalWeldings
# Load the .mat file
#with h5py.File('./VAE/preprocessed2.mat', 'r') as mat_file:

#with h5py.File('./VAE/preprocessed_normalized.mat', 'r') as mat_file:
with h5py.File('./VAE/preprocessed(theta_ma+bias,1iteration).mat', 'r') as mat_file:
    # Access the 'bc_dict' group
    bc_dict_group = mat_file['bc_dict']

    # Initialize a 3D NumPy array to store theta
    conformalWeldings = np.empty((len(bc_dict_group), 1, 200), dtype=np.float32)  # Use float32 instead of complex128

    # Iterate over the fields (e.g., 'Case00_12', 'Case00_13', etc.)
    for i, field_name in enumerate(bc_dict_group):
        case_group = bc_dict_group[field_name]
        xq = case_group['x'][:]  # Load the 'x' dataset into a structured array
        yq = case_group['y'][:]  # Load the 'y' dataset into a structured array
        theta = case_group['theta'][:]  # Load the 'theta' dataset into a structured array
        theta = np.insert(theta, 0, 0)
        theta = np.diff(theta) # Use diff between theta to train
        theta_ma = case_group['theta_ma'][:]  # Load the 'theta_ma' dataset into a structured array
        theta_ma = np.insert(theta_ma, 0, 0)
        theta_ma = np.diff(theta_ma)
        bias = case_group['bias'][:]  # Load the 'bias' dataset into a structured array
        
        conformalWeldings[i, :, :100] = theta # Correspond theta and bias together
        conformalWeldings[i, :, -100:] = bias 

        # Store the real and imaginary parts separately
        # conformalWeldings[i, 0, :] = np.abs(xq['real'])
        # conformalWeldings[i, 1, :] = np.abs(xq['imag'])
        
        #conformalWeldings[i, 0, :] = np.abs(yq['real']).astype(float)
        #conformalWeldings[i, 1, :] = np.abs(yq['imag']).astype(float)
        
        #conformalWeldings[i, 0, :] = yq['real'].astype(float)
        #conformalWeldings[i, 1, :] = yq['imag'].astype(float)

        

# Assuming you have your 777x2x1000 data stored in a NumPy array named 'conformalWeldings'
#conformalWeldings = conformalWeldings[:10, :, :]
conformalWeldings = torch.tensor(conformalWeldings).to(DEVICE) 



# Create a DataLoader for your dataset
train_data = TensorDataset(conformalWeldings)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Start Training
for epoch in range(NUM_EPOCHS):
    adjust_learning_rate(optimizer, epoch)
    loop = tqdm(enumerate(train_loader))
    
    reconstruction_losses = []  # To store reconstruction losses
    reconstruction_diff_losses = []  # To store reconstruction diff losses
    kl_div_losses = []         # To store KL divergence losses

    for i, batch in loop:
        for x in batch:
            # Forward Pass
            x = x.to(DEVICE, dtype=torch.float32).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(x)

            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            reconstruction_diff_loss = loss_fn(torch.diff(x_reconstructed, dim=1), torch.diff(x, dim=1))
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop
            #loss = model.alpha * reconstruction_loss + (1-model.alpha) * kl_div
            #loss = kl_div
            #loss = reconstruction_loss
            loss = reconstruction_loss + 0.2 * kl_div
            #loss = reconstruction_loss + 0.1*reconstruction_diff_loss + 0.1 * kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

            # Append losses to the lists
            reconstruction_losses.append(reconstruction_loss.item())
            reconstruction_diff_losses.append(reconstruction_diff_loss.item())
            kl_div_losses.append(kl_div.item())

    # Calculate and print average losses for this epoch
    avg_reconstruction_loss = sum(reconstruction_losses) / len(reconstruction_losses)
    avg_reconstruction_diff_loss = sum(reconstruction_diff_losses) / len(reconstruction_diff_losses)
    avg_kl_div_loss = sum(kl_div_losses) / len(kl_div_losses)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Alpha: {model.alpha.item():.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.1e}")
    print(f"Reconstruction Loss: {avg_reconstruction_loss:.4f}, Reconstruction Diff Loss: {avg_reconstruction_diff_loss:.4f}, KL Divergence Loss: {avg_kl_div_loss:.4f}")




model = model.to("mps")

#torch.save(model, './VAE_bias.pth')
torch.save(model, './VAE_theta+bias1.pth')