import numpy as np
from sgdml.predict import GDMLPredict
import argparse 
from sgdml.utils import io

parser = argparse.ArgumentParser(description="Predict loop for sGDML")

parser.add_argument("-d","--dataset", type=str, help="dataset file path")
parser.add_argument("-s","--save", type=str, help="model save path")
parser.add_argument("-n","--n_train", default=200, type=int)

model = np.load('sGDML/saved/glycine_200.npz')
gdml = GDMLPredict(model)

r,_ = io.read_xyz('Datasets/Glycine.xyz')
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
