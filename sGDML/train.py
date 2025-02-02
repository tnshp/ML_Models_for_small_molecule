import sys
import numpy as np
import argparse 
from sgdml.train import GDMLTrain

parser = argparse.ArgumentParser(description="Training loop for sGDML")

parser.add_argument("-d","--dataset", type=str, help="dataset file path")
parser.add_argument("-s","--save", type=str, help="model save path")
parser.add_argument("-n","--n_train", default=200, type=int)

# Parse arguments
args = parser.parse_args()

dataset = np.load(args.dataset)
n_train = args.n_train

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

np.savez_compressed(args.save, **model)
print("Model Saved!!!")
