import os
import torch
import torchmetrics
import pytorch_lightning as pl

import sys
sys.path.insert(0, 'src\schnetpack')

import schnetpack as spk
import schnetpack.representation as rep
import schnetpack.atomistic as atm
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

# Import the necessary components from schnetpack.nn
from schnetpack.nn import cutoff, radial

# Load dataset, focusing only on forces
db_file = 'Azobenzene.db'  # Ensure the correct path
dataset = ASEAtomsData(db_file, load_properties=['forces'])

# Check dataset length
print(f"Dataset length: {len(dataset)}")

# Use the file path directly for AtomsDataModule
custom_data = spk.data.AtomsDataModule(
    datapath=db_file,  # Pass the file path, not the dataset object
    batch_size=24,  # Ensure this matches with the batch size in DataLoader
    distance_unit='Ang',
    property_units={'forces': 'kcal/mol/Ang'},
    num_train=int(len(dataset) * 0.8),  # Properly cast to int
    num_val=int(len(dataset) * 0.2),  # Properly cast to int
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.CastTo32()
    ],
    num_workers=0,  # Number of worker subprocesses for data loading
    pin_memory=True,  # Pin memory for faster data transfer between CPU and GPU
    split_file=None  # Ensure that no split file is used
)

# Prepare and set up data
custom_data.prepare_data()
custom_data.setup()

# Access data loaders
train_loader = custom_data.train_dataloader()
val_loader = custom_data.val_dataloader()

# Check the length of the dataset to confirm the split
print(f"Training dataset length: {len(train_loader.dataset)}")
print(f"Validation dataset length: {len(val_loader.dataset)}")

# Define SchNet representation and model
cutoff_fn = cutoff.CosineCutoff(cutoff=5.0)
radial_basis = radial.GaussianRBF(cutoff=5.0, n_rbf=50)

schnet = rep.SchNet(
    n_atom_basis=128,
    n_interactions=6,
    radial_basis=radial_basis,
    cutoff_fn=cutoff_fn
)

# Pairwise distance module
pairwise_distance = atm.PairwiseDistances()

# Output module only for forces (no energy)
# Modify the forces module to predict energy
pred_forces = atm.Forces(energy_key='energy', force_key='forces')
pred_energy = atm.Atomwise(n_in=128, output_key='energy')

# Assemble model with both energy and forces
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],  # Include both energy and forces
    postprocessors=[trn.CastTo64()]
)

# Define loss function and metric only for forces
output_forces = spk.task.ModelOutput(
    name='forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,  # 100% focus on forces
    metrics={"MAE": torchmetrics.MeanAbsoluteError()}
)

print("Output forces: \n", output_forces)

# Assemble task, focusing only on forces
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-3}
)

# Set up logger and checkpointing
forcetut = "./forcetut"
logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
callbacks = [
    ModelCheckpoint(
        dirpath=os.path.join(forcetut, "checkpoints"),
        filename="best_inference_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
]

# Use GPU if available, otherwise fall back to CPU
device = 'gpu' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for training.")

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=10,  # Adjust epochs as needed
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    devices=1 if torch.cuda.is_available() else None,  # Use one GPU
)

trainer.fit(task, train_loader, val_loader)

# Save the trained model
torch.save(task, "trained_model.pth")
print("Model saved successfully.")
