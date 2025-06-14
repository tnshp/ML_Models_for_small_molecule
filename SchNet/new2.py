import torch 
from ase.io import read
from ase import Atoms
import schnetpack.transform as trn
import os
import schnetpack as spk

model_path = './saved/train_NQ_n1000.pth'
device = 'cuda'
best_model = torch.load(model_path, map_location=device)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
)
reactant = read('./Simulation/NQ/opt_nbd.xyz')
product = read('./Simulation/NQ/opt_qc.xyz')
# convert atoms to SchNetPack inputs and perform prediction
inputs = converter(product)
results = best_model(inputs)

print(results)

# print(reactant.get_positions())
# R = torch.Tensor(reactant.get_positions())
# print(model(R))