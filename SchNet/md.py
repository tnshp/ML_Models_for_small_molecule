from schnetpack.md import Simulator, System, LangevinIntegrator
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.md.simulation_hooks import LangevinThermostat, DataLogger
from schnetpack.transform import ASENeighborList
from schnetpack import Properties

import torch
from ase.io import read
import torch
import os
import argparse


parser = argparse.ArgumentParser(description="Testing loop for sGDML")
parser.add_argument("-m","--model", type=str, help="model file path")
parser.add_argument("-i","--initial_struct", type=str, help="Inital structure file path in xyz format")
parser.add_argument("-log","--log_dir", type=str, help="logging directory")

args = parser.parse_args()
# 1. Load trained model and set device
model_path = args.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model = model.to(device)

# 2. Configure neighbor list for MD
cutoff = 5.0  # Å (should match model training)
cutoff_shell = 2.0  # Å
md_neighborlist = NeighborListMD(
    cutoff,
    cutoff_shell,
    ASENeighborList,
    device=device
)

# 3. Set up calculator WITHOUT device argument
md_calculator = SchNetPackCalculator(
    model,
    md_neighborlist,
    energy_key=Properties.energy,
    force_key=Properties.forces,
    required_properties=[Properties.energy, Properties.forces]
)

# 4. Initialize system and simulator
initial_structure = read(args.initial_struct)
md_system = System()
md_system.load_molecules(initial_structure)

integrator = LangevinIntegrator(
    time_step=0.5,  # fs
    temperature=300,  # K
    friction=0.01  # 1/fs
)
thermostat = LangevinThermostat(
    temperature=300,  # K
    time_constant=100  # fs
)

logger = DataLogger(
    "md_logs.hdf5",
    buffer_size=100,
    data_streams=[
        md_system.get_energy_stream(),
        md_system.get_position_stream()
    ]
)

md_simulator = Simulator(
    system=md_system,
    integrator=integrator,
    calculator=md_calculator,
    simulation_hooks=[thermostat, logger],
    step_count=10000
)

md_simulator.simulate()
