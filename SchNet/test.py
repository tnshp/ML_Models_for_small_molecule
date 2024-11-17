import os
import torch
import numpy as np
from ase import Atoms
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model_path = "trained_model_8.pth"
best_model = torch.load(model_path, map_location=device)
best_model.eval()  # Set model to evaluation mode

# Set up the converter for converting ASE Atoms objects to SchNetPack inputs
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
)

# Load the dataset (make sure the paths and property names are correct)
db_path = "md\Azobenzene.db"
dataset = ASEAtomsData(db_path, load_properties=['forces'])

# Initialize lists to store the true and predicted values
all_forces = []
forces = []

# Loop through all the entries in the dataset
for i in range(len(dataset)):
    if i%1000==0:
        print(f"{i} completed")
    structure = dataset[i]  # Get the structure from the dataset

    # Create an ASE Atoms object from the dataset entry
    atoms = Atoms(
        numbers=structure[spk.properties.Z],  # Atomic numbers (element type)
        positions=structure[spk.properties.R]  # Atomic positions
    )

    # Convert the Atoms object into SchNetPack-compatible inputs
    inputs = converter(atoms)

    # Run the model on the inputs and get the prediction
    results = best_model(inputs)

    # Extract the predicted forces and energy

    # For forces, do the same
    predicted_force = results["forces"].detach().cpu().numpy()
    true_force = structure["forces"]  # No need for .cpu().numpy() here, it's already a NumPy array

    # Append the true and predicted values to the lists
    all_forces.append(true_force)  # Append directly as it's already a NumPy array
    forces.append(predicted_force)  # Append directly as it's already a NumPy arr

# Convert the lists to NumPy arrays
all_forces = np.array(all_forces)
predicted_forces = np.array(forces)

# Flatten the forces to 2D (number of atoms x 3 for each atom's force vector)
print(all_forces)
print(predicted_forces)
print(all_forces.shape)
print(predicted_forces.shape)
# all_forces = all_forces.reshape(-1, 3)
# predicted_forces = predicted_forces.reshape(-1, 3)

# Compute the RMSE and MAE for forces and energy

# Function to calculate RMSE
def rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

# Function to calculate MAE
def mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))

# Compute RMSE and MAE for forces
rmse_forces = rmse(all_forces, predicted_forces)
mae_forces = mae(all_forces, predicted_forces)

# Compute RMSE and MAE for energy
# Print out the results
print(f"RMSE for Forces: {rmse_forces}")
print(f"MAE for Forces: {mae_forces}")
