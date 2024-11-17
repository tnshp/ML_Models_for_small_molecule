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
from schnetpack.transform import RemoveOffsets

# Load dataset, focusing only on energy
db_file = 'md/Azorot.db'  # Ensure the correct path
dataset = ASEAtomsData(db_file, load_properties=['energy'])

# Check dataset length
print(f"Dataset length: {len(dataset)}")

remove_offsets = RemoveOffsets(
    property='energy',
    remove_mean=True,           # Remove the mean
    remove_atomrefs=False,      # No single-atom reference used
    is_extensive=True,          # Since energy is extensive
    zmax=100
)

# Use the file path directly for AtomsDataModule
custom_data = spk.data.AtomsDataModule(
    datapath=db_file,
    batch_size=24,
    distance_unit='Ang',
    property_units={'energy': 'kcal/mol'},
    # num_train=int(len(dataset) * 0.8),
    # num_val=int(len(dataset) * 0.2),
    num_train=400,
    num_val=80,
    transforms=[
        trn.ASENeighborList(cutoff=5.0),
        trn.CastTo32(),
        remove_offsets
    ],
    num_workers=0,
    pin_memory=False,
    split_file=None
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

# Output module for energy prediction
pred_energy = atm.Atomwise(n_in=128, output_key='energy')

# Assemble model for energy prediction only
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy],
    postprocessors=[trn.CastTo64()]
)

# Define loss function and metric for energy
output_energy = spk.task.ModelOutput(
    name='energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()}
)

print("Output energy: \n", output_energy)

# Assemble task for energy prediction
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-3}
)

# Set up logger and checkpointing
forcetut = "./forcetut"
logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
callbacks = [
    ModelCheckpoint(
        dirpath=os.path.join(forcetut, "checkpoints"),
        filename="best_energy_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
]

# Configure Trainer
trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=10,  # Adjust epochs as needed
    accelerator='cpu'
)

# Train the model
trainer.fit(task, train_loader, val_loader)

# Save the trained model
torch.save(task, "trained_rot_4.pth")
print("Energy model saved successfully.")
