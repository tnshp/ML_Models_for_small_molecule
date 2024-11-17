import sys
import numpy as np
from sgdml.train import GDMLTrain

dataset = np.load('/home/AnirbanMondal_grp/23110035/sgdml/Azobenzene_rotation.npz')
n_train = 200

energy_mean = np.mean(dataset['E'])
energy_std = np.std(dataset['E'])

# Create a mutable dictionary from the read-only dataset
modified_data = {key: dataset[key] for key in dataset}

# Normalize energy
modified_data['E'] = (modified_data['E'] - energy_mean) / energy_std
gdml_train = GDMLTrain()

task = gdml_train.create_task(modified_data, n_train,\
        valid_dataset=dataset, n_valid=40,\
        sig=20, lam=1e-10)

model = gdml_train.train(task)

np.savez_compressed('m_azorot_2.npz', **model)
print("Model Saved!!!")
