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
from pytorch_lightning.callbacks import ModelCheckpoint

from schnetpack.nn import cutoff, radial

def parse_args():
    parser = argparse.ArgumentParser(description="SchNetPack Force Prediction")
    parser.add_argument("--output_dir", type=str,  help="Directory for output files")
    parser.add_argument("--db_file", type=str,  help="Path to the database file")
    parser.add_argument("--model_save_path", type=str, default="trained_model.pth", help="Path to save the trained model")
    
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff distance for interactions")
    parser.add_argument("--n_atom_basis", type=int, default=128, help="Number of features to describe atomic environments")
    parser.add_argument("--n_interactions", type=int, default=6, help="Number of interaction blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of training epochs")
    
    return parser.parse_args()

def main(args):
    # Load dataset, focusing only on forces
    dataset = ASEAtomsData(args.db_file, load_properties=['forces'])
    print(f"Dataset length: {len(dataset)}")

    # Use the file path directly for AtomsDataModule
    custom_data = spk.data.AtomsDataModule(
        datapath=args.db_file,
        batch_size=args.batch_size,
        distance_unit='Ang',
        property_units={'forces':'kcal/mol/Ang'},
        num_train=int(len(dataset)*0.8),
        num_val=int(len(dataset)*0.2),
        transforms=[
            trn.ASENeighborList(cutoff=args.cutoff),
            trn.CastTo32()
        ],
        num_workers=0,
        pin_memory=False,
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
        loss_weight=1.0,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )

    print("Output forces: \n", output_forces)

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_forces],
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
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator='cpu',
    )

    trainer.fit(task, train_loader, val_loader)

    torch.save(task, args.model_save_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
