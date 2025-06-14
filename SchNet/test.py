import os 
import torch
import numpy as np
from ase import Atoms
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
import argparse

# Function to calculate RMSE
def rmse(true_values, predicted_values):
    val = (true_values - predicted_values) ** 2
    #remove outliers 
    val = np.where(np.abs(val) < 300, val, 0)  # Set outliers to 0
    val = np.sqrt(np.mean(val))
    return val

# Function to calculate MAE
def mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate SchNet model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the dataset database')
    args = parser.parse_args()

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    best_model = torch.load(args.model_path, map_location=device)
    best_model.eval()  # Set model to evaluation mode

    # Set up the converter for converting ASE Atoms objects to SchNetPack inputs
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    ) 

    # Load the dataset (make sure the paths and property names are correct)
    dataset = ASEAtomsData(args.db_path, load_properties=['forces', 'energy'])
    # dataset = dataset.subset(range(1000, len(dataset)))
    #sample 4000 random points from the dataset
    val = np.random.choice(len(dataset), size=4000, replace=False).tolist()  # Convert to Python list
    dataset = dataset.subset(val)  # Sample the first 4000 entries
    dataset.subset
    # Initialize lists to store the true and predicted values
    all_forces = []
    all_energy = [] 

    forces = []
    energy = []

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
        predicted_energy = results["energy"].detach().cpu().numpy()
        predicted_force = results["forces"].detach().cpu().numpy()
        true_energy = structure["energy"]  
        true_force = structure["forces"]  

        # Append the true and predicted values to the lists
        all_forces.append(true_force)  # Append directly as it's already a NumPy array
        all_energy.append(true_energy)

        forces.append(predicted_force)  # Append directly as it's already a NumPy arr
        energy.append(predicted_energy)

    # Convert the lists to NumPy arrays
    all_forces = np.array(all_forces)
    predicted_forces = np.array(forces)

    all_energy = np.array(all_energy)
    predicted_energy =np.array(energy)

    # Flatten the forces to 2D (number of atoms x 3 for each atom's force vector)
    # all_forces = all_forces.reshape(-1, 3)
    # predicted_forces = predicted_forces.reshape(-1, 3)

    # Compute the RMSE and MAE for forces and energy

    

    # Compute RMSE and MAE for forces
    rmse_forces = rmse(all_forces, predicted_forces)
    mae_forces = mae(all_forces, predicted_forces)
    rmse_energy = rmse(all_energy, predicted_energy)
    mae_energy = mae(all_energy, predicted_energy)
    
    # Compute RMSE and MAE for energy
    # Print out the results
    print(f"RMSE for Energy: {rmse_energy}")
    print(f"MAE for Energy: {mae_energy}")
    print(f"RMSE for Forces: {rmse_forces}")
    print(f"MAE for Forces: {mae_forces}")

if __name__ == "__main__":
    main()
