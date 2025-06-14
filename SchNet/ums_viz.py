import numpy as np
import matplotlib.pyplot as plt
from schnetpack import units as spk_units
from schnetpack.md.data import HDF5Loader
from ase.io import read, write
import os 

ums_workdir='ums/2025-06-12_11-26-37'
dirs = os.listdir(ums_workdir)
dirs = [d for d in dirs if os.path.isdir(os.path.join(ums_workdir, d))]
# Filter out directories that do not start with 'window_'
dirs = [d for d in dirs if d.startswith('window_')]
# Sort directories to ensure consistent order
n_centers = len(dirs)

print(f"Number of centers: {n_centers}")

for i in range(n_centers):
    md_workdir = os.path.join(ums_workdir, f"window_{i}")
    if not os.path.exists(md_workdir):
        continue
    print(f"Processing {md_workdir}")
    log_file= os.path.join(md_workdir, "simulation.hdf5")
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist, skipping.")
        continue
    data = HDF5Loader(log_file)

    for prop in data.properties:
        print(prop)


    energies_calculator = data.get_property('energy', atomistic=False)
    energies_system = data.get_potential_energy()

    # Check the overall shape
    print("Shape:", energies_system.shape)

    # Get the time axis
    time_axis = np.arange(data.entries) * data.time_step / spk_units.fs  # in fs

    # Convert the system potential energy from internal units (kJ/mol) to kcal/mol
    energies_system *= spk_units.convert_units("eV/mol", "kcal/mol")
    # Plot the energies
    plt.figure()
    # plt.plot(time_axis, energies_system, label="E$_\mathrm{pot}$ (System)")
    plt.plot(time_axis, energies_calculator, label="E$_\mathrm{pot}$ (Logger)", ls="--")
    plt.ylabel("E [eV/mol]")
    plt.xlabel("t [fs]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(md_workdir, "energies.png"), dpi=300)
    # plt.show()

    md_atoms = data.convert_to_atoms()

    # write list of Atoms to XYZ file
    write(
        os.path.join(md_workdir, "trajectory.xyz"),
        md_atoms,
        format="xyz"
    )