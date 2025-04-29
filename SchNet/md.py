import torch
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, kB
from schnetpack.interfaces import SpkCalculator 
import argparse
from ase.io import read

parser = argparse.ArgumentParser(description="Testing loop for sGDML")
parser.add_argument("-m","--model", type=str, help="model file path")
parser.add_argument("-i","--initial_struct", type=str, help="Inital structure file path in xyz format")
parser.add_argument("-log","--log_dir", type=str, help="logging directory")

args = parser.parse_args()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.model, map_location='cpu')
model.eval()

# Define your initial atomic configuration
# Replace this with your actual structure: NBD molecule
atoms = read(args.initial_struct)
# atoms = Atoms('C7H8', positions=[[...], [...], ...])  # Fill in with your coordinates

# Attach the SchNet model as a calculator
calc = SpkCalculator(model, energy='energy', forces='forces')
atoms.set_calculator(calc)

# Set the initial temperature (e.g., 300 K)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Set up the MD simulation (Verlet integrator)
dyn = VelocityVerlet(atoms, dt=0.5 * fs)

# Define a simple logger
def printenergy(a=atoms):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    print(f'Epot = {epot:.3f} eV, Ekin = {ekin:.3f} eV, Etot = {epot+ekin:.3f} eV')

dyn.attach(printenergy, interval=10)

# Run MD for 1000 steps
dyn.run(1000)
