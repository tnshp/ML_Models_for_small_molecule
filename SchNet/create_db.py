import numpy as np
import os
from ase import Atoms
from schnetpack.data import ASEAtomsData

# Load the NPZ file
data = np.load('md\Azobenzene.npz')

# Extract arrays from the NPZ file
positions = data["positions"]
atomic_numbers = data["atomic_numbers"]
forces = data["forces"]
energies = data["energies"]

print(forces)
print(energies)
# Prepare data for ASEAtomsData
atoms_list = []
property_list = []

for i in range(len(positions)):
    # Create Atoms object for each system
    atoms = Atoms(positions=positions[i], numbers=atomic_numbers[i])
    
    # Ensure forces and energies are in the correct format
    energy = np.array(energies[i])
    # if isinstance(energy, np.ndarray) and energy.ndim == 1 and energy.size == 1:
    #     energy = energy[0]  # Convert back to scalar if needed by ASE
    properties = {'energy': energy, 'forces': forces[i]}
    
    atoms_list.append(atoms)
    property_list.append(properties)

print('Properties:', property_list[0])
# Define the database file path
db_file = 'md/Azobenzene.db'

# Remove the existing database file if it exists
if os.path.exists(db_file):
    os.remove(db_file)

# Create a new SchNetPack database
new_dataset = ASEAtomsData.create(
    db_file,
    distance_unit='Ang',
    property_unit_dict={'energy': 'kcal/mol', 'forces': 'kcal/mol/Ang'}
)

# Add systems to the database
new_dataset.add_systems(
    property_list=property_list,
    atoms_list=atoms_list
)

print(f"Database created and saved to {db_file}")

