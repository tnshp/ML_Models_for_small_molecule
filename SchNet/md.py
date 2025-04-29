from schnetpack.md import Simulator, System
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack import AtomsData
from ase.io import read
import os
import argparse

parser = argparse.ArgumentParser(description="Testing loop for sGDML")

parser.add_argument("-m","--model", type=str, help="model file path")
parser.add_argument("-i","--initial_struct", type=str, help="Inital structure file path in xyz format")
parser.add_argument("-log","--log_dir", type=str, help="logging directory")
# Parse arguments 
args = parser.parse_args()


# Load your trained model
model_path = args.model  # Replace with your model path
md_calculator = SchNetPackCalculator(
    model_path,
    device="cuda",  # Use "cpu" if no GPU
    energy_key="energy",
    force_key="forces",
    required_properties=["energy", "forces"]
)

#intitial structure
initial_structure = read(args.initial_struct)

from schnetpack.md import LangevinSimulator
from schnetpack.md.integrators import LangevinIntegrator
from schnetpack.md.simulation_hooks import Checkpoint, DataLogger

# System setup
md_system = System()
md_system.load_molecules(initial_structure)

# Integrator (NVT ensemble)
integrator = LangevinIntegrator(
    time_step=0.5,  # fs
    temperature=300,  # K
    friction=0.01  # 1/fs
)

# Simulation hooks (logging)
logger = DataLogger(
    os.path.join(args.log_dir, "md_logs.hdf5"),
    buffer_size=100,
    data_streams=[
        md_system.get_energy_stream(),
        md_system.get_position_stream()
    ]
)

# Simulator
md_simulator = LangevinSimulator(
    system=md_system,
    integrator=integrator,
    calculator=md_calculator,
    simulation_hooks=[logger],
    step_count=10000  # Total MD steps
)

md_simulator.simulate()