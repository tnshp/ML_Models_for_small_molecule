import os
import argparse
import torch

import schnetpack as spk
from schnetpack.md import System, Simulator, MaxwellBoltzmannInit, UniformInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack import properties
from schnetpack.md.simulation_hooks import SimulationHook, LangevinThermostat, callback_hooks
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList
from schnetpack import units as spk_units
from schnetpack.md.utils import NormalModeTransformer

from schnetpack.md.calculators import MDCalculator
from ase.constraints import FixBondLengths
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm
import numpy as np
from datetime import datetime

class BiasCalculator(MDCalculator):
    def __init__(self, center_system: System, center, k_spring, refference,  energy_unit, position_unit):
        super().__init__(
            required_properties=["energy", "forces"],
            force_key="forces",
            energy_key="energy",
            energy_unit=energy_unit,
            position_unit=position_unit
        )
        self.k_spring = k_spring
        
        self.center_system = center_system
        self.center = center
        # self.center = self.center_system.positions  # Get the first replica's position

        if not isinstance(self.center, torch.Tensor):
            self.center = torch.tensor(center, dtype=torch.float32) 
        
        self.refference = refference
        if not isinstance(self.refference, torch.Tensor):
            self.refference = torch.tensor(refference, dtype=torch.float32) 
        
        self.internal2positions = spk_units.convert_units(
            spk_units.length, "Angstrom"
        )
        self.positions2internal = spk_units.convert_units(
            "Angstrom", spk_units.length
        )
        self.center = self.center * self.positions2internal

    def calculate(self, system: System):
        positions = system.positions # Shape: [n_replicas, n_molecules*n_atoms, 3]

        # 1. Compute potential energy (your custom function)
        energy = self._custom_energy(positions) 
        forces = self._custom_forces(positions)  # Shape: [n_replicas, n_molecules*n_atoms, 3]
        
        # Store results in system
        self.results = {
            "energy": energy,
            "forces": forces
        }
        # self._update_system(system)
    
    def _custom_energy(self, positions):
        #bias potential 
        bias_potential = 0.5 * self.k_spring * torch.sum((positions - self.center)**2, dim=-1) 
        return bias_potential.sum()

    def _custom_forces(self, positions):
        # Compute forces based on the bias potential
        forces =  self.k_spring * (self.center - positions )
        return forces
    
class CompositeCalculator(MDCalculator):
    def __init__(self, bias_calc, schnet_calc, energy_unit=None, position_unit=None):
        super().__init__(
            required_properties=["energy", "forces"],
            force_key="forces",
            energy_key="energy",
            energy_unit=energy_unit,
            position_unit=position_unit
        )
        self.bias_calc = bias_calc
        self.schnet_calc = schnet_calc

    def calculate(self, system):
        total_energy = 0
        total_forces = 0
        bias = 0
        
        self.bias_calc.calculate(system)
        bias =  self.bias_calc.results["energy"]
        total_energy += bias

        if self.bias_calc.results["forces"].shape != system.forces.shape:
            self.bias_calc.results["forces"] = self.bias_calc.results["forces"].view(system.forces.shape)
        total_forces += self.bias_calc.results["forces"]

        inputs = self.schnet_calc._generate_input(system)
        result = self.schnet_calc.model(inputs)
        if result['forces'].shape != system.forces.shape:
            result['forces'] = result['forces'].view(system.forces.shape)
        # print(result['energy'].shape, result['forces'].shape)
        # print(total_energy.shape, total_forces.shape)
        total_energy += result['energy'].sum()  # Sum over replicas
        total_forces += result['forces']

        self.results = {
            "energy": total_energy,
            "forces": total_forces,
            "bias"  : bias
        }

        self._update_system(system)
        
    
def interpolate_structures(reactant, product, n_centers, output_prefix="center"):
    assert len(reactant) == len(product), "Reactant and product must have same number of atoms"
   
    # Get Cartesian coordinates
    R = reactant.get_positions()
    P = product.get_positions()
    
    centers = []
    for i in range(n_centers):
        t = i / (n_centers - 1)
        coords = (1 - t) * R + t * P  # Linear interpolation[2][5]
        centers.append(coords)
    return centers

if __name__ == "__main__":
    
    reactant_path = "Simulation/NQ/opt_qc.xyz"
    product_path  = "Simulation/NQ/opt_nbd.xyz"
    n_centers = 10  # Number of centers to generate
    output_prefix = "center"  # Prefix for output files

    k_spring = 1000
    n_steps = 5000
    system_temperature = 300
    time_step = 0.1         # fs
    bath_temperature = 300  # K
    time_constant = 100     # fs
    buffer_size = 100
    cutoff = 5.0  # Angstrom (units used in model)
    cutoff_shell = 2.0  # Angstrom

    reactant = read(reactant_path)
    product  = read(product_path)


    r_mean = np.mean(reactant.get_positions(),  axis=0)
    reactant.set_positions(reactant.get_positions() - r_mean)

    p_mean = np.mean(product.get_positions(),  axis=0)
    product.set_positions(product.get_positions() - p_mean)
    # centers = interpolate_structures(reactant, product, n_centers, output_prefix)
    centers = interpolate_structures(reactant, product, n_centers, output_prefix)

    #save centers:
    save_dir = "ums"
    center_atoms = []

    for i, center in enumerate(centers):
        center_atoms.append(
            Atoms(
                symbols=reactant.get_chemical_symbols(),
                positions=center,
                cell=reactant.get_cell(),
                pbc=reactant.get_pbc()
            )
        )

    # center_file = os.path.join(save_dir, f"interpolation.xyz")
    # write(center_file, center_atoms, format="xyz")

    md_workdir = "ums"

    md_system = System()
    md_system.load_molecules(
        reactant,
        1,
        position_unit_input="Angstrom"
    )

    

    # Set up the initializer
    md_initializer = UniformInit(
        system_temperature,
        remove_center_of_mass=True,
        remove_translation=True,
        remove_rotation=True,
    )

    # Initialize the system momenta
    md_initializer.initialize_system(md_system)


    md_integrator = VelocityVerlet(time_step)

    
    md_neighborlist = NeighborListMD(
        cutoff,
        cutoff_shell,
        ASENeighborList,
    )

    model_path = "saved/train_NQ_n1000.pth"  # Path to trained model

    eV_to_kcalmol  = 23.0605
    schnet_calculator = SchNetPackCalculator(
        model_path,  # path to stored model
        "forces",
        eV_to_kcalmol,
        "Angstrom",  # length units
        md_neighborlist,  # neighbor list
        energy_key="energy",  # name of potential energies
        required_properties=[],  # additional properties extracted from the model
    )

    bias_calculator = BiasCalculator(
        md_system, 
        center=centers[5],  # Use the first center as the bias center
        k_spring=k_spring  ,  # Spring constant for the bias potential
        refference=centers[0],  # Reference point for bias
        energy_unit="kcal/mol",
        position_unit="Angstrom"
    )

    md_calculator = CompositeCalculator(
        bias_calculator,
        schnet_calculator,
        energy_unit=eV_to_kcalmol,
        position_unit="Angstrom"
    )

    langevin = LangevinThermostat(bath_temperature, time_constant)
    
    simulation_hooks = [
        langevin
    ]

    log_file = os.path.join(md_workdir, "simulation.hdf5")
    #delete log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)

    data_streams = [
        callback_hooks.MoleculeStream(store_velocities=True),
        callback_hooks.PropertyStream(target_properties=[properties.energy]),
    ]

    
    file_logger = callback_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=5,  # logging frequency
        precision=32,  # floating point precision used in hdf5 database
    )


    simulation_hooks.append(file_logger)
    chk_file = os.path.join(md_workdir, 'simulation.chk')

    # Create the checkpoint logger
    checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)
    simulation_hooks.append(checkpoint)

    # check if a GPU is available and use a CPU otherwise
    if torch.cuda.is_available():
        md_device = "cuda"
    else:
        md_device = "cpu"
    print(f'running on {md_device}')
 
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

    positions2internal = spk_units.unit2internal("Angstrom")
    print("spk units:", positions2internal)
    

    print("Total number of steps:", md_simulator.step)
    md_simulator.simulate(n_steps)
