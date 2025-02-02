import numpy as np
from sgdml.predict import GDMLPredict
from sgdml.utils import io
import argparse

parser = argparse.ArgumentParser(description="Training loop for sGDML")

parser.add_argument("-m","--model", type=str, help="model file path")
parser.add_argument("-d","--dataset", type=str, help="dataset file path")

# Parse arguments
args = parser.parse_args()

# Load the pre-trained GDML model
model = np.load(args.model)
gdml = GDMLPredict(model)

r, metadata = io.read_xyz(args.dataset)

def parse_xyz_file(file_path):
    positions = []
    atomic_numbers = []
    forces = []
    energies = []
    
    atom_to_number = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    }
    
    with open(args.dataset, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        
        num_atoms = int(lines[i].strip())
        i += 1
        
        energy = float(lines[i].strip())
        energies.append([energy])  # Wrap energy in a list to make it a 1D array
        i += 1
        
        atom_nums = []
        pos = []
        force = []
        for _ in range(num_atoms):
            parts = lines[i].strip().split()
            atom_symbol = parts[0]
            if atom_symbol in atom_to_number:
                atom_num = atom_to_number[atom_symbol]
            else:
                raise ValueError(f"Unknown atomic symbol: {atom_symbol}")
            
            atom_nums.append(atom_num)
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
            force.append([float(parts[4]), float(parts[5]), float(parts[6])])
            i += 1
        
        atomic_numbers.append(atom_nums)
        positions.append(pos)
        forces.append(force)
    
    return np.array(positions), np.array(atomic_numbers), np.array(forces), np.array(energies)
        
positions, atomic_numbers, forces, energies = parse_xyz_file(args.dataset)

energy_mean = np.mean(energies)
energy_std = np.std(energies)

# Normalize the energies from the test set
energies = (energies - energy_mean) / energy_std

# Predict energy and forces using the GDML model
predicted_energy, predicted_forces = gdml.predict(r)


if predicted_forces.shape != forces.shape:
    predicted_forces = predicted_forces.reshape(-1, 24, 3)

# Function to calculate RMSE
def rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

# Function to calculate MAE
def mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))

# Calculate RMSE and MAE for energy
rmse_energy = rmse(energies, np.array(predicted_energy))
mae_energy = mae(energies, np.array(predicted_energy))

# Calculate MSE and MAE for forces
print(forces)
print(predicted_forces)
rmse_forces = rmse(forces, predicted_forces)
mae_forces = mae(forces, predicted_forces)

# Print the RMSE and MAE results
print(f"RMSE for Energy: {rmse_energy:.6f}")
print(f"MAE for Energy: {mae_energy:.6f}")
print(f"RMSE for Forces: {rmse_forces:.6f}")
print(f"MAE for Forces: {mae_forces:.6f}")
