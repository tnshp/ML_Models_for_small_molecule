import numpy as np
from sgdml.predict import GDMLPredict
from sgdml.utils import io

model = np.load('m_Azobenzene_inversion.npz')
gdml = GDMLPredict(model)

r,_ = io.read_xyz('Azobenzene_inversion.xyz')
e,f = gdml.predict(r)

with open("output.txt", "w") as file:
    file.write("Energies:\n")
    for energy in e:
        file.write(f"{energy}\n")
        
    file.write("\nForces:\n")
    for force_vector in f:
        file.write(" ".join(map(str, force_vector)) + "\n")

print(r.shape)
print(e.shape)
print(f.shape)
print(r)
print(e)
print(f)
