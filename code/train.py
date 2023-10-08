import os
import json
import wandb
import torch
import argparse

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from header import DICOMDataset, DICOMDataModule, ResNet50Module

def main(args):
    # pl.seed_everything(RANDOM_STATE_CONSTANT)
    # Initialize data module
    filepath = os.path.join(os.getcwd(), args.json_file)
    dm = DICOMDataModule(filepath, batch_size=args.batch_size)
    dm.setup()

    # Initialize model
    model = ResNet50Module(max_epochs=args.max_epochs)

    # Initialize logger
    wandb_logger = WandbLogger(
        project="hfhs_cnn", 
        name="resnet50_rin",
        log_model="all",
        save_dir="..",
        config = {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "device": args.device,
            "num_classes": 2,
            "rin_backbone": True,
        }
    )

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="resnet50-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=args.device,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        fast_dev_run=args.dev_run,
        deterministic=False,
    )

    # print("something")
    # print("hi")
    # Train the model
    trainer.fit(model, dm)

    # Test the model
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    RANDOM_STATE_CONSTANT = 1026
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="Path to JSON file containing DICOM data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train for")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (default: cpu)")
    parser.add_argument("--dev_run", action="store_true", help="Run a development test")
    args = parser.parse_args()

    # Set up environment variables
    # os.environ["WANDB_MODE"] = "disabled"
    os.environ["NO_CUDA"] = "1"

    main(args)