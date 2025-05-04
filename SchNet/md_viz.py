import numpy as np
import matplotlib.pyplot as plt
from schnetpack import units as spk_units
from schnetpack.md.data import HDF5Loader
from ase.io import read, write
import os 

md_workdir='md_run'
log_file='md_run/simulation.hdf5'
data = HDF5Loader(log_file)

for prop in data.properties:
    print(prop)



# Get the energy logged via PropertiesStream
energies_calculator = data.get_property('energy', atomistic=False)
# Get potential energies stored in the MD system
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
plt.ylabel("E [kcal/mol]")
plt.xlabel("t [fs]")
plt.legend()
plt.tight_layout()
plt.show()

md_atoms = data.convert_to_atoms()

# write list of Atoms to XYZ file
write(
    os.path.join(md_workdir, "trajectory.xyz"),
    md_atoms,
    format="xyz"
)