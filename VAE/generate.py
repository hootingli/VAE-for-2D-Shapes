import torch
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder  # Assuming you have a VAE model defined in 'model.py'
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import scipy.io
import h5py
import random


# Configuration
model = torch.load('./VAE_theta+bias1.pth') # Use trained and saved model here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
INPUT_DIM = 200  # Adjusted input dimension
H_DIM = 200
Z_DIM = 4

def generate(num_examples=1, mu_vals=None, sigma_vals=None, save_path='./output.mat'):
    if mu_vals is None:
        mu_vals = [0] * 4
        #mu_vals = [random.uniform(-0.2, 0.2) for _ in range(4)]
    if sigma_vals is None:
        sigma_vals = [1.2] * 4

    generated_data = []
    mu = torch.tensor(mu_vals, dtype=torch.float).to(DEVICE)
    sigma = torch.tensor(sigma_vals, dtype=torch.float).to(DEVICE)
    
    for i in range(num_examples):
        epsilon = torch.randn_like(sigma)
        #mu = torch.tensor([random.uniform(-5, 5) for _ in range(4)], dtype=torch.float).to(DEVICE)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(1, 1, INPUT_DIM)  # Adjusted output shape
        #out = out * 10
        generated_data.append(out)
        #print(z)

    # Convert the generated data to a numpy array
    generated_data = torch.cat(generated_data, dim=0).cpu().detach().numpy()

    # Save the generated data as a .mat file
    scipy.io.savemat(save_path, {'generated_data': generated_data})

generate(num_examples=100, save_path='./output_theta+bias1.mat')
#generate(num_examples=20, mu_vals=[1.2971e-05, 1.4011e-04], save_path='./output2.mat')

'''
# Load Data
with h5py.File('./VAE/preprocessed_normalized.mat', 'r') as mat_file:
    # Access the 'bc_dict' group
    bc_dict_group = mat_file['bc_dict']

    # Initialize a 3D NumPy array to store theta
    conformalWeldings = np.empty((len(bc_dict_group), 1, 100), dtype=np.float32)  # Use float32 instead of complex128

    # Iterate over the fields (e.g., 'Case00_12', 'Case00_13', etc.)
    for i, field_name in enumerate(bc_dict_group):
        case_group = bc_dict_group[field_name]
        xq = case_group['x'][:]  # Load the 'x' dataset into a structured array
        yq = case_group['y'][:]  # Load the 'y' dataset into a structured array
        theta = case_group['theta'][:]  # Load the 'theta' dataset into a structured array

        conformalWeldings[i, :, :] = theta / 10
        # Store the real and imaginary parts separately
        # conformalWeldings[i, 0, :] = np.abs(xq['real'])
        # conformalWeldings[i, 1, :] = np.abs(xq['imag'])
        
        #conformalWeldings[i, 0, :] = np.abs(yq['real']).astype(float)
        #conformalWeldings[i, 1, :] = np.abs(yq['imag']).astype(float)
        
        #conformalWeldings[i, 0, :] = yq['real'].astype(float)
        #conformalWeldings[i, 1, :] = yq['imag'].astype(float)

conformalWeldings = torch.tensor(conformalWeldings).to(DEVICE)
'''


'''
def inference(digit, num_examples=1, save_path='./output.mat'):
    generated_data = []
    with torch.no_grad():
        mu, sigma = model.encode(conformalWeldings[digit].view(1, INPUT_DIM))
        encodings_digit = [(mu, sigma) for _ in range(num_examples)]

    print(f'mu = {mu}')
    #print(f'sigma = {sigma}')

    for encodings in encodings_digit:
        mu, sigma = encodings
        epsilon = torch.randn_like(sigma)
        # epsilon = 0
        z = mu + sigma * epsilon
        z = mu
        out = model.decode(z)
        out = out.view(1, 1, 100)  # Adjusted output shape
        generated_data.append(out)

    # Convert the generated data to a numpy array
    generated_data = torch.cat(generated_data, dim=0).cpu().detach().numpy()

    # Save the generated data as a .mat file
    #scipy.io.savemat(save_path, {'generated_data': generated_data})
'''
'''
inference(1)
inference(59)
inference(180)
inference(324)
inference(516)
inference(687)
'''