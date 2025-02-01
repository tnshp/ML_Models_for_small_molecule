import numpy as np
import argparse 

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
    
    with open(file_path, 'r') as file:
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


parser = argparse.ArgumentParser(description="A simple script that to convert xyz to npz format")

parser.add_argument("-x","--xyz_file_path", type=str, help="xyz file path")
parser.add_argument("-n","--npz_file_path", type=str, help="npz file path")

# Parse arguments
args = parser.parse_args()


# Parse the XYZ file
positions, atomic_numbers, forces, energies = parse_xyz_file(args.xyz_file_path)

print(forces)
print(energies)

# Save data to NPZ file
np.savez(args.npz_file_path, positions=positions, atomic_numbers=atomic_numbers, forces=forces, energies=energies)

print(f"Data saved to {args.npz_file_path}")

