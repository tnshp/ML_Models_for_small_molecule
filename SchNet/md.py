import os
import torch
import schnetpack as spk
from ase import Atoms
from schnetpack.md import UniformInit
from schnetpack.md import System
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack import properties
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import LangevinThermostat
from ase.constraints import FixBondLengths
from ase.io import read
import argparse
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser(description="MD run for schnet")
parser.add_argument("-m", "--model", type=str, help="model file path")
parser.add_argument("-i", "--initial_struct", type=str, help="Initial structure file path in xyz format")
parser.add_argument("-dir", "--md_workdir", type=str, help="Logging directory")
parser.add_argument("--temperature", default=300, type=float, help="Logging directory")
parser.add_argument("--n_steps", default=1000, type=int, help="Logging directory")
parser.add_argument("--time_step", default=0.1, type=float, help="time step in femtosecond")
args = parser.parse_args()

md_workdir = args.md_workdir
# Gnerate a directory of not present
if not os.path.exists(md_workdir):
    os.mkdir(md_workdir)
else:
    import shutil
    shutil.rmtree(md_workdir)
    os.mkdir(md_workdir)
# Get the parent directory of SchNetPack
spk_path = os.path.abspath(os.path.join(os.path.dirname(spk.__file__), '../..'))

# Load model and structure
model_path = args.model
molecule_path = args.initial_struct

# Load atoms with ASE
molecule = read(molecule_path)
c_indices = [a.index for a in molecule if a.symbol == 'C']
h_indices = [a.index for a in molecule if a.symbol == 'H']

# Find nearest C for each H using KDTree
c_pos = molecule.positions[c_indices]
tree = cKDTree(c_pos)
bond_pairs = []
for h in h_indices:
    _, idx = tree.query(molecule.positions[h])
    c = c_indices[idx]
    bond_pairs.append((c, h))
# Number of molecular replicas
# molecule.set_constraint(FixBondLengths(bond_pairs))

n_replicas = 1

# Create system instance and load molecule
md_system = System()
md_system.load_molecules(
    molecule,
    n_replicas,
    position_unit_input="Angstrom"
)

system_temperature = args.temperature # Kelvin

# Set up the initializer
md_initializer = UniformInit(
    system_temperature,
    remove_center_of_mass=True,
    remove_translation=True,
    remove_rotation=True,
)

# Initialize the system momenta
# md_initializer.initialize_system(md_system)


time_step = args.time_step # fs

# Set up the integrator
md_integrator = VelocityVerlet(time_step)
# md_integrator = Rattle(time_step, bonds=bond_pairs)

from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList

# set cutoff and buffer region
cutoff = 5.0  # Angstrom (units used in model)
cutoff_shell = 2.0  # Angstrom

# initialize neighbor list for MD using the ASENeighborlist as basis
md_neighborlist = NeighborListMD(
    cutoff,
    cutoff_shell,
    ASENeighborList,
)


eV_to_kcalmol  = 23.0605
md_calculator = SchNetPackCalculator(
    model_path,  # path to stored model
    "forces",
    eV_to_kcalmol,
    # 'eV/mol',
    "Angstrom",  # length units
    md_neighborlist,  # neighbor list
    energy_key="energy",  # name of potential energies
    required_properties=[],  # additional properties extracted from the model
)




# Set temperature and thermostat constant
bath_temperature = args.temperature  # K
time_constant = 100  # fs

# Initialize the thermostat
langevin = LangevinThermostat(bath_temperature, time_constant)

simulation_hooks = [
    langevin
]

from schnetpack.md.simulation_hooks import callback_hooks

# Path to database
log_file = os.path.join(md_workdir, "simulation.hdf5")

# Size of the buffer
buffer_size = 100

# Set up data streams to store positions, momenta and the energy
data_streams = [
    callback_hooks.MoleculeStream(store_velocities=True),
    callback_hooks.PropertyStream(target_properties=[properties.energy]),
]

# Create the file logger
file_logger = callback_hooks.FileLogger(
    log_file,
    buffer_size,
    data_streams=data_streams,
    every_n_steps=1,  # logging frequency
    precision=32,  # floating point precision used in hdf5 database
)

# Update the simulation hooks
simulation_hooks.append(file_logger)

#Set the path to the checkpoint file
chk_file = os.path.join(md_workdir, 'simulation.chk')

# Create the checkpoint logger
checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)

# Update the simulation hooks
simulation_hooks.append(checkpoint)

# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    md_device = "cuda"
else:
    md_device = "cpu"

# use single precision
md_precision = torch.float32

md_simulator = Simulator(
    md_system,
    md_integrator,
    md_calculator,
    simulator_hooks=simulation_hooks
)
# set precision
md_simulator = md_simulator.to(md_precision)
# move everything to target device
md_simulator = md_simulator.to(md_device)

n_steps = args.n_steps

md_simulator.simulate(n_steps)
print("Total number of steps:", md_simulator.step)