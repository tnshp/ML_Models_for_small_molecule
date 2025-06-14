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
from schnetpack.md.calculators import MDCalculator
from schnetpack import units as spk_units

from ase import Atoms
from ase.io import read
from tqdm import tqdm
import wham 
import numpy as np
from datetime import datetime

class UmbrellaHook(SimulationHook):
    def __init__(
        self,
        rc_function,
        center,
        refference,
        k=100.0,  # kcal/mol/unit²
        device='cuda'
    ):
        """
        reaction_coordinate: Function computing RC from positions (returns tensor)
        center: Target value for reaction coordinate (tensor)
        k: Harmonic force constant
        """
        super().__init__()
        self.rc_function = rc_function
        self.center = center
        if type(self.center) != torch.tensor:
            self.center = torch.tensor(center).to(device)

        self.refference = refference
        if type(self.refference) != torch.tensor:
            self.refference = torch.tensor(refference).to(device)
        
        self.k = k
        self.observables = {'rc': [], 'bias': [], 'energy': []}
        self.internal2positions = spk_units.convert_units(
            spk_units.length, "Angstrom"
        )
    
    
    def on_step_finalize(self, simulator):
        #stroe the rc values 
        position  = simulator.system.positions
        position = position * self.internal2positions

        rc = self.rc_function(position, self.refference) # Compute reaction coordinate
        energy = simulator.calculator.results["energy"]  # Get energy from calculator
        bias = simulator.calculator.results["bias"] 
        self._store_observables(rc, bias, energy)
    
    def _store_observables(self,rc, bias, energy):
        self.observables["rc"].append(rc.item())
        self.observables["bias"].append(bias.item())
        self.observables["energy"].append(energy.item())

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
            required_properties=["energy", "forces", "bias"], 
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
            "bias": bias
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


def distance_rc(positions, ref_positions):
    # positions and ref_positions are torch tensors of shape (N, 3)
    if type(positions) == torch.Tensor:
        diff = positions - ref_positions
        rmsd = torch.sqrt(torch.sum(diff**2))
        return rmsd
    if type(positions) == np.ndarray:
        diff = positions - ref_positions
        rmsd = np.sqrt(np.sum(diff**2))
        return rmsd

def get_logger(workdir, idx):
    #logging 
    data_streams = [
        callback_hooks.MoleculeStream(store_velocities=True),
        callback_hooks.PropertyStream(target_properties=[properties.energy]),
    ]
    buffer_size = 100  # Size of the buffer for logging

    md_dir = os.path.join(workdir, f'window_{idx}')
    if not os.path.exists(md_dir):
        os.makedirs(md_dir)
    log_file = os.path.join(md_dir, f'simulation.hdf5')
    file_logger = callback_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=5,  # logging frequency
        precision=32,  # floating point precision used in hdf5 database
    )
    return file_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MD run for schnet")
    parser.add_argument("-m", "--model", type=str, help="model file path")
    parser.add_argument("-r", "--reactant", type=str, help="Initial structure file path in xyz format")
    parser.add_argument("-p", "--product", type=str, help="final structure file path in xyz format")
    parser.add_argument("-n", "--n_centers", default=40, type=int, help="Number of windows")
    parser.add_argument("-k", "--k_spring", default=500, type=int, help="Number of windows")
    # parser.add_argument("-dir", "--md_workdir", type=str, help="Logging directory")
    parser.add_argument("--temperature", default=400, type=float, help="Logging directory")
    parser.add_argument("--n_steps", default=1000, type=int, help="Logging directory")
    parser.add_argument("--time_step", default=0.1, type=float, help="time step in femtosecond")
    parser.add_argument("--damping", default=100, type=float, help="time step in femtosecond")
    args = parser.parse_args()

    # md_workdir = args.md_workdir 
    model_path = args.model
    reactant_path = args.reactant
    product_path = args.product
    time_step = args.time_step # fs
    system_temperature = args.temperature # Kelvin
    bath_temperature = args.temperature  # K
    time_constant = args.damping  # fs
    n_steps = args.n_steps
    k_spring = args.k_spring
    cutoff = 5.0  # Angstrom (units used in model)
    cutoff_shell = 2.0  # Angstrom
    n_replicas = 1

    #reactant -> product
    reactant = read(reactant_path)
    product  = read(product_path)
    
    r_mean = np.mean(reactant.get_positions(),  axis=0)
    reactant.set_positions(reactant.get_positions() - r_mean)

    p_mean = np.mean(product.get_positions(),  axis=0)
    product.set_positions(product.get_positions() - p_mean)

    centers = interpolate_structures(reactant, product, args.n_centers)

    md_system = System()
    md_system.load_molecules(
        reactant,
        n_replicas,
        position_unit_input="Angstrom"
    )

    # Set up the initializer
    md_initializer = UniformInit(
        system_temperature,
        remove_center_of_mass=True,
        remove_translation=True,
        remove_rotation=True,
    )
    # md_initializer.initialize_system(md_system)

    md_integrator = VelocityVerlet(time_step)

    # initialize neighbor list for MD using the ASENeighborlist as basis
    md_neighborlist = NeighborListMD(
        cutoff,
        cutoff_shell,
        ASENeighborList,
    )

    eV_to_kcalmol  = 23.0605
    schnet_calc = SchNetPackCalculator(
        model_path,  # path to stored model
        "forces",
        eV_to_kcalmol,
        # 'eV/mol',
        "Angstrom",  # length units
        md_neighborlist,  # neighbor list
        energy_key="energy",  # name of potential energies
        required_properties=[],  # additional properties extracted from the model
    )
    
    # Initialize the thermostat
    langevin = LangevinThermostat(bath_temperature, time_constant)

    if torch.cuda.is_available():
        md_device = "cuda"
    else:
        md_device = "cpu"

    md_precision = torch.float32
     
    #starting window md
    trajectories = []
    all_rc_window = []
    all_bias_window = []
    all_energy_window = []

    
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    workdir = os.path.join('ums',  timestamp)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    for idx, center in tqdm(enumerate(centers)):
        bias_calc = BiasCalculator(
            md_system, 
            center=center,  # Use the reactant positions as the bias center
            k_spring=k_spring,  # Spring constant for the bias potential
            refference=reactant.get_positions(),  # Reference point for bias
            energy_unit="kcal/mol",
            position_unit="Angstrom"
        )
        md_calculator = CompositeCalculator(
            bias_calc,
            schnet_calc,
            energy_unit=eV_to_kcalmol,
            position_unit="Angstrom"
        )
        umbrella_hook = UmbrellaHook(
            rc_function=distance_rc,
            center=center,  # Use the current center for this window
            refference=reactant.get_positions(),  # Reference positions (e.g., reactant positions)
            k=args.k_spring,  # Spring constant for the bias potential
            device=md_device
        )

        file_logger = get_logger(workdir, idx)
        
        simulation_hooks = [
            langevin,
            umbrella_hook,
            file_logger,
        ]

        md_simulator = Simulator(
            md_system,
            md_integrator,
            md_calculator,
            simulator_hooks=simulation_hooks
        )
        
        md_simulator = md_simulator.to(md_precision)
        md_simulator = md_simulator.to(md_device)

        md_simulator.simulate(n_steps)
         
        all_rc_window.append(umbrella_hook.observables['rc'])
        all_bias_window.append(umbrella_hook.observables['bias'])
        all_energy_window.append(umbrella_hook.observables['energy'])


    # After running all windows and collecting rc values:
    mean_rcs = [np.mean(rc) for rc in all_rc_window]  # <xi> for each window
   
    center_rcs = [distance_rc(center, reactant.get_positions()) for center in centers]  
    center_rcs = np.array(center_rcs)  # reaction coordinate for each window center    # centers = np.array(centers)  # window centers
    k_spring = float(args.k_spring)  # force constant (assume same for all windows)
    mean_forces = -k_spring * (np.array(mean_rcs) - center_rcs)
    
    # Sort by center (just in case)
    sort_idx = np.argsort(center_rcs)
    centers_sorted = center_rcs[sort_idx]
    mean_forces_sorted = mean_forces[sort_idx]

    # Integrate mean force to get PMF (potential of mean force)
    # Use the trapezoidal rule for numerical integration
    pmf = np.zeros_like(centers_sorted)
    for i in range(1, len(centers_sorted)):
        dx = centers_sorted[i] - centers_sorted[i-1]
        pmf[i] = pmf[i-1] + (mean_forces_sorted[i] + mean_forces_sorted[i-1]) * dx

    # Normalize PMF (set minimum to zero)
    pmf -= pmf.min()

    # Plotting
    import matplotlib.pyplot as plt
    plt.plot(centers_sorted, pmf, label='PMF')
    plt.legend()
    plt.xlabel('Reaction Coordinate (Angstrom)')
    plt.ylabel('Free Energy (kcal/mol)')
    plt.title('PMF via Umbrella Integration')
    plt.savefig(os.path.join(workdir, 'pmf_umbrella_integration.png'))
    plt.clf()

    plt.plot(centers_sorted, mean_forces_sorted, label='mean force', color='orange')
    plt.legend()
    plt.xlabel('Reaction Coordinate (Angstrom)')
    plt.ylabel('mean force (kcal/mol/angstrom)')
    plt.title('Mean force via Umbrella Integration')
    plt.savefig(os.path.join(workdir, 'mean_force_umbrella_integration.png'))
    plt.clf()

    #save the rc, bias and energy values for each window
    np.savez(
        os.path.join(workdir, 'umbrella_data.npz'),
        rc=np.array(all_rc_window),
        bias=np.array(all_bias_window),
        energy=np.array(all_energy_window),
        centers=np.array(centers_sorted),
        pmf=np.array(pmf),
        mean_forces=np.array(mean_forces_sorted)
    )
    print(f"Simulation completed. Results saved in {workdir}")






