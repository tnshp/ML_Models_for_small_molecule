import os
import torch
import torchmetrics
import pytorch_lightning as pl
import argparse
import sys

sys.path.insert(0, 'src\schnetpack')

import schnetpack as spk
import schnetpack.representation as rep
import schnetpack.atomistic as atm
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from schnetpack.nn import cutoff, radial

def parse_args():
    parser = argparse.ArgumentParser(description="SchNetPack Force Prediction")
    parser.add_argument("--db_file", type=str, required=True, help="Path to the database file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff distance for interactions")
    parser.add_argument("--n_atom_basis", type=int, default=128, help="Number of features to describe atomic environments")
    parser.add_argument("--n_interactions", type=int, default=6, help="Number of interaction blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of training epochs")
    parser.add_argument("--model_save_path", type=str, default="trained_model.pth", help="Path to save the trained model")
    parser.add_argument("--num_train", type=int, default=1000, help="Number of samples to use for training")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (-1 for all available)")
    parser.add_argument("--early_stopping", action="store_true", help="enable early stopping")
    return parser.parse_args()

def main(args):
    # Load dataset, focusing only on forces
    dataset = ASEAtomsData(args.db_file, load_properties=['forces'])
    print(f"Total dataset length: {len(dataset)}")

    # Set num_train and calculate num_val
    args.num_train = min(args.num_train, len(dataset) - 1)  # Ensure at least one sample for validation
    args.num_val = len(dataset) - args.num_train
    print(f"Using {args.num_train} training sample and {args.num_val} validation for {args.max_epochs} epochs")

    # Use the file path directly for AtomsDataModule
    custom_data = spk.data.AtomsDataModule(
        datapath=args.db_file,
        batch_size=args.batch_size,
        distance_unit='Ang',
        property_units={'forces':'kcal/mol/Ang'},
        num_train=args.num_train,
        num_val=args.num_val,
        transforms=[
            trn.ASENeighborList(cutoff=args.cutoff),
            trn.CastTo32()
        ],
        num_workers=4,  # Increased for better performance
        pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
        split_file=None
    )

    custom_data.prepare_data()
    custom_data.setup()

    train_loader = custom_data.train_dataloader()
    val_loader = custom_data.val_dataloader()

    print(f"Training dataset length: {len(train_loader.dataset)}")
    print(f"Validation dataset length: {len(val_loader.dataset)}")

    cutoff_fn = cutoff.CosineCutoff(cutoff=args.cutoff)
    radial_basis = radial.GaussianRBF(cutoff=args.cutoff, n_rbf=50)

    schnet = rep.SchNet(
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn
    )

    pairwise_distance = atm.PairwiseDistances()
    pred_forces = atm.Forces(energy_key='energy', force_key='forces')
    pred_energy = atm.Atomwise(n_in=args.n_atom_basis, output_key='energy')

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[trn.CastTo64()]
    )

    output_forces = spk.task.ModelOutput(
        name='forces',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.7,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )
    output_energy = spk.task.ModelOutput(
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.3,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": args.lr}
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best_inference_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        ),
    ]
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=5)
        )

    # Determine GPU usage
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    
    if args.gpus > 0:
        accelerator = 'gpu'
        devices = args.gpus
    else:
        accelerator = 'cpu'
        devices = 1

    print(f"Using accelerator: {accelerator}, devices: {devices}")

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy='ddp' if devices and devices > 1 else 'auto',  # Use DDP for multi-GPU training
        enable_progress_bar=False
    )
    

    trainer.fit(task, train_loader, val_loader)

    
    torch.save(task, os.path.join(args.output_dir, args.model_save_path))
    print("Model saved successfully.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
