import h5py
import pandas as pd
import numpy as np
# Load the .mat file
with h5py.File('./VAE/preprocessed.mat', 'r') as mat_file:
    # Access the 'bc_dict' group
    bc_dict_group = mat_file['bc_dict']

    # Initialize a 3D NumPy array to store xq and yq arrays
    data_array = np.empty((len(bc_dict_group), 2, 1000), dtype=np.complex128)

    # Iterate over the fields (e.g., 'Case00_12', 'Case00_13', etc.)
    for i, field_name in enumerate(bc_dict_group):
        case_group = bc_dict_group[field_name]
        xq = case_group['x'][:]  # Load the 'x' dataset into a structured array
        yq = case_group['y'][:]  # Load the 'y' dataset into a structured array

        # Create complex arrays from the real and imaginary parts
        xq_complex = xq['real'] + 1j * xq['imag']
        yq_complex = yq['real'] + 1j * yq['imag']

        # Store the complex xq and yq arrays in the 3D array
        data_array[i, 0, :] = xq_complex
        data_array[i, 1, :] = yq_complex