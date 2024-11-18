import sys
import numpy as np
from sgdml.train import GDMLTrain

# Load the dataset
dataset = np.load('/home/sgdml/Azobenzene_rotation.npz')
n_train = 200  # Increased training set size

# Normalize the energy values
energy_mean = np.mean(dataset['E'])
energy_std = np.std(dataset['E'])
modified_data = {key: dataset[key] for key in dataset}
modified_data['E'] = (modified_data['E'] - energy_mean) / energy_std

# Initialize the trainer
gdml_train = GDMLTrain()

# Tune hyperparameters
sig = 15  # Smaller value might be better for rapid energy changes
lam = 1e-8  # Slightly stronger regularization

# Create a task focusing on energy optimization
task = gdml_train.create_task(
    modified_data,
    n_train,
    valid_dataset=dataset,
    n_valid=40,
    sig=sig,
    lam=lam,
    use_E_cstr=True,
    use_E=True  # Focus only on energy
)

# Train the model
model = gdml_train.train(task)

# Save the model
np.savez_compressed('m_azorot_optimized_2.npz', **model)
print("Optimized Model Saved!")