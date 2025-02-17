import numpy as np
import os
import argparse
from ase import Atoms
from schnetpack.data import ASEAtomsData

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create SchNetPack database from NPZ file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input NPZ file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output database file')
    args = parser.parse_args()

    # Load the NPZ file
    data = np.load(args.input_file)

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
        properties = {'energy': energy, 'forces': forces[i]}
        
        atoms_list.append(atoms)
        property_list.append(properties)

    print('Properties:', property_list[0])

    # Remove the existing database file if it exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    # Create a new SchNetPack database
    new_dataset = ASEAtomsData.create(
        args.output_file,
        distance_unit='Ang',
        property_unit_dict={'energy': 'kcal/mol', 'forces': 'kcal/mol/Ang'}
    )

    # Add systems to the database
    new_dataset.add_systems(
        property_list=property_list,
        atoms_list=atoms_list
    )

    print(f"Database created and saved to {args.output_file}")

if __name__ == "__main__":
    main()
