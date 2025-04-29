import argparse
import torch
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
from ase.io.trajectory import Trajectory
from schnetpack.interfaces import SpkCalculator
from schnetpack.transform import ASENeighborList
import os

# Argument parser
parser = argparse.ArgumentParser(description="Testing loop for sGDML")
parser.add_argument("-m", "--model", type=str, help="model file path")
parser.add_argument("-i", "--initial_struct", type=str, help="Initial structure file path in xyz format")
parser.add_argument("-log", "--log_dir", type=str, help="Logging directory")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained model
model = torch.load(args.model, map_location=device)
model = model.to(device)
model.eval()

# Load initial structure
atoms = read(args.initial_struct)
atoms.set_pbc([False, False, False])  # Disable periodic boundary conditions

# Create neighbor list object
neighbor_list = ASENeighborList(cutoff=5.0)

# Attach SchNet calculator
calc = SpkCalculator(
    model=model,
    neighbor_list=neighbor_list,
    energy="energy",
    forces="forces"
)
atoms.set_calculator(calc)

# Initialize temperature
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Set up logger
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
traj = Trajectory(os.path.join(args.log_dir, "trajectory.traj"), "w", atoms)

# Integrator
dyn = VelocityVerlet(atoms, dt=0.5 * fs)

# Logging callback
def log_energy(a=atoms):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    print(f"Step: {dyn.nsteps:4d} | Epot: {epot:.4f} eV | Ekin: {ekin:.4f} eV | Etot: {epot + ekin:.4f} eV")
    traj.write(a)

dyn.attach(log_energy, interval=10)

# Run simulation
print("Starting MD simulation...")
dyn.run(1000)
print("MD simulation complete.")
traj.close()
