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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from schnetpack.nn import cutoff, radial

class ForceEnergyTask(spk.task.AtomisticTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize metrics for validation
        self.val_forces_mae = torchmetrics.MeanAbsoluteError()
        self.val_forces_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_energy_mae = torchmetrics.MeanAbsoluteError()
        self.val_energy_rmse = torchmetrics.MeanSquaredError(squared=False)
        
        # Initialize metrics for testing
        self.test_forces_mae = torchmetrics.MeanAbsoluteError()
        self.test_forces_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_energy_mae = torchmetrics.MeanAbsoluteError()
        self.test_energy_rmse = torchmetrics.MeanSquaredError(squared=False)

    def validation_step(self, batch, batch_idx):
        results = super().validation_step(batch, batch_idx)
        preds = self.model(batch)
        
        # Update validation metrics
        self.val_forces_mae(preds['forces'], batch['forces'])
        self.val_forces_rmse(preds['forces'], batch['forces'])
        self.val_energy_mae(preds['energy'], batch['energy'])
        self.val_energy_rmse(preds['energy'], batch['energy'])
        
        return results

    def on_validation_epoch_end(self):
        # Log validation metrics
        self.log('val_forces_mae', self.val_forces_mae.compute(), prog_bar=True)
        self.log('val_forces_rmse', self.val_forces_rmse.compute())
        self.log('val_energy_mae', self.val_energy_mae.compute(), prog_bar=True)
        self.log('val_energy_rmse', self.val_energy_rmse.compute())
        
        # Reset metrics
        self.val_forces_mae.reset()
        self.val_forces_rmse.reset()
        self.val_energy_mae.reset()
        self.val_energy_rmse.reset()

    def test_step(self, batch, batch_idx):
        results = super().test_step(batch, batch_idx)
        preds = self.model(batch)
        
        # Update test metrics
        self.test_forces_mae(preds['forces'], batch['forces'])
        self.test_forces_rmse(preds['forces'], batch['forces'])
        self.test_energy_mae(preds['energy'], batch['energy'])
        self.test_energy_rmse(preds['energy'], batch['energy'])
        
        return results

    def on_test_epoch_end(self):
        # Log test metrics
        self.log('test_forces_mae', self.test_forces_mae.compute(), prog_bar=True)
        self.log('test_forces_rmse', self.test_forces_rmse.compute())
        self.log('test_energy_mae', self.test_energy_mae.compute(), prog_bar=True)
        self.log('test_energy_rmse', self.test_energy_rmse.compute())
        
        # Print formatted metrics
        print(f"\nTest Metrics:")
        print(f"Forces MAE: {self.test_forces_mae.compute():.4f}")
        print(f"Forces RMSE: {self.test_forces_rmse.compute():.4f}")
        print(f"Energy MAE: {self.test_energy_mae.compute():.4f}")
        print(f"Energy RMSE: {self.test_energy_rmse.compute():.4f}")
        
        # Reset metrics
        self.test_forces_mae.reset()
        self.test_forces_rmse.reset()
        self.test_energy_mae.reset()
        self.test_energy_rmse.reset()

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
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of dataset to use for testing")
    return parser.parse_args()

def main(args):
    # Load dataset with both forces and energy
    dataset = ASEAtomsData(args.db_file, load_properties=['forces', 'energy'])
    print(f"Total dataset length: {len(dataset)}")

    # Calculate splits
    total = len(dataset)
    args.num_train = min(args.num_train, total - 2)
    num_val_test = total - args.num_train
    args.num_val = int(num_val_test * (1 - args.test_ratio))
    args.num_test = num_val_test - args.num_val

    print(f"Using {args.num_train} train, {args.num_val} val, and {args.num_test} test samples")

    # Configure data module
    custom_data = spk.data.AtomsDataModule(
        datapath=args.db_file,
        batch_size=args.batch_size,
        distance_unit='Ang',
        property_units={'forces': 'kcal/mol/Ang', 'energy': 'kcal/mol'},
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        transforms=[
            trn.ASENeighborList(cutoff=args.cutoff),
            trn.CastTo32()
        ],
        num_workers=4,
        pin_memory=True,
        split_file=None
    )

    custom_data.prepare_data()
    custom_data.setup()

    # Model configuration
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

    # Configure outputs with metrics
    output_forces = spk.task.ModelOutput(
        name='forces',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.7,
        metrics={
            'MAE': torchmetrics.MeanAbsoluteError(),
            'RMSE': torchmetrics.MeanSquaredError(squared=False)
        }
    )

    output_energy = spk.task.ModelOutput(
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.3,
        metrics={
            'MAE': torchmetrics.MeanAbsoluteError(),
            'RMSE': torchmetrics.MeanSquaredError(squared=False)
        }
    )

    # Configure training task
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_forces, output_energy],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": args.lr}
    )

    # Configure trainer
    logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
    metric_tracker = MetricTracker()
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        ),
        metric_tracker,
        CustomProgressBar()
    ]

    # GPU configuration
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    
    accelerator = 'gpu' if args.gpus > 0 else 'cpu'
    devices = args.gpus if args.gpus > 0 else 1
    strategy = 'ddp' if args.gpus > 1 else 'auto'

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        enable_progress_bar=True
    )

    # Training and testing
    trainer.fit(task, custom_data)
    
    # Run test on best model
    print("\nStarting test phase...")
    trainer.test(dataloaders=custom_data.test_dataloader(), ckpt_path='best')

    # Save model
    torch.save(task, os.path.join(args.output_dir, args.model_save_path))
    print("Model saved successfully.")

if __name__ == "__main__":
    args = parse_args()
    main(args)

