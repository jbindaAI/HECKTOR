# IMPORT PACKAGES
from Dataset import CropData
from Model import TwoHeadsBeast
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from torchvision.transforms import v2
#from lifelines.utils import concordance_index
#import wandb
import argparse
#from monai import transforms as T
def main():
    # DEFINE PARAMETERS
    epochs = 40
    milestones = [15, 30]
    lr_schedule = "fixed"
    channels = ["CT", "PT"]
    select_transforms = ["crop"]
    tb_logs_path = "/home/dzban112/HECKTOR/tb_logs"
    checkpoints_path = "/home/dzban112/HECKTOR/checkpoints"
    MODEL_NR = 1
    
    # CREATE THE PARSER
    parser = argparse.ArgumentParser(description="Train an end-to-end model")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for the model"
    )
    parser.add_argument(
        "--tie_method",
        type=str,
        default="breslow",
        help='Tie method can be either "breslow" or "efron"',
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=5,
        help="Patience for learning rate annealing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--val_split",
        type=int,
        default=42,
        help="Random seed for validation split",
    )
    parser.add_argument(
        "--trainable_layers",
        type=int,
        default=10,
        help="When fine-tuning a pretrained model, how many layers to unfreeze",
    )
    
    args = parser.parse_args()
    
    learning_rate = args.learning_rate
    tie_method = args.tie_method
    patience = args.patience
    batch_size = args.batch_size
    val_split = args.val_split
    trainable_layers = args.trainable_layers
    
    print(f"Learning rate: {learning_rate}")
    print(f"Tie method: {tie_method}")
    print(f"Deform probability: {deform_prob}")
    print(f"Intensity probability: {intensity_prob}")
    print(f"Patience: {patience}")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {val_split}")
    print(f"Number of trainable layers: {trainable_layers}")
    
    torch.set_float32_matmul_precision("medium")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # DEFINE AUGMENTATIONS
    train_transforms = T.Compose(v2.Resize([224, 224]))
    test_transforms = T.Compose(v2.Resize([224, 224]))
    
    # DEFINE DATA SPLITS
    all_train_data = CropData(train=True, transform=train_transforms, channels=channels)
    # Set a seed for the random number generator and split data into training and validational.
    torch.manual_seed(val_split)
    train_dataset, val_dataset = random_split(all_train_data, [0.8, 0.2])
    val_dataset.transform = test_transforms
    # Make an instance of test data
    test_dataset = CropData(train=False, transform=test_transforms, channels=channels)
    
    # DATA LOADERS
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    
    # MAKE AN INSTANCE OF THE MODEL
    model = TwoHeadsBeast(
        milestones=milestones,
        learning_rate=learning_rate,
        tie_method=tie_method,
        lr_schedule=lr_schedule,
        patience=patience,
        trainable_layers=trainable_layers,
        dropout=dropout
    )
    
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=checkpoints_path,
        filename="best-{epoch:02d}",
    )
    
    
    # TRAINING
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=epochs,
        logger=TensorBoardLogger(tb_logs_path, name=f"CT_PET_model_{MODEL_NR}"),
        callbacks=[checkpoint_callback, lr_monitor],
        strategy="ddp_find_unused_parameters_true",
        log_every_n_steps=6,
    )
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    del trainer
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
