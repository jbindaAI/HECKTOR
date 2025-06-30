# IMPORT PACKAGES
from lightning_modules.data_module import HECKTOR_DataModule
from lightning_modules.model_module import HECKTOR_Model

import os
import math
import torch
import wandb
import pickle
from typing import Literal, Union
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


# HYPERPARAMETERS:
## Run Metadata:
MODEL_NR: int = 1
## General setup
MODEL_TYPE: Literal["vanilla_vit", "deit_vit", "dino_vit", "dino_v2_reg", "clip"] = "vanilla_vit"
LOCAL: bool = True
SAVE_TOP_CKPTS: int = 3
WANDB_PROJECT: str = "HECKTORv2"
DO_CV: bool = True # Whether perform 5-Fold Cross-Validation.
FULL_TRAINING: bool = True # Whether train model on the whole training set, without CV. 
## Data setup
MODALITY: Literal["CT", "PET", "Merged"] = "CT"
BBOX_OR_CENTROID: Literal["bbox", "centroid"] = "bbox"
## Training setup
EPOCHS: int = 50
BATCH_SIZE: int = 16
LR: float = 3e-4
TIE_METHOD: Literal["breslow", "efron"] = "breslow"
BCKB_DROPOUT: float = 0.2
WEIGHT_DECAY: float = 0.01
WARMUP_ITERS: int = 100
NUM_WORKERS: int = 8

if LOCAL:
    DATA_PATH="/home/jakub/research/HECKTOR/Data"
    checkpoints_path="/home/jakub/research/HECKTOR/ckpt"
else:
    DATA_PATH="/home/dzban112/HECKTOR/Data/"
    checkpoints_path="/home/dzban112/HECKTOR/ckpt/"

def train_model(fold: Union[int, Literal["all"]] = 1):
    """
    If fold is integer, then model is trained on a specified CV fold and evaluated on a corresponding validational fold.
    When fold == "all", then model is trained using all training examples without evaluation on separate validational fold.
    """
    # Getting value of training steps:
    with open(os.path.join(DATA_PATH,"filtered_labels",f"train_labels/train_fold_{fold}.pkl"), "rb") as f:
        n_train_examples = len(pickle.load(f))
        steps_per_epoch = math.ceil(n_train_examples/BATCH_SIZE)
        total_steps = steps_per_epoch*EPOCHS

    # add a checkpoint callback that saves the model with the highest validation concordance index.
    checkpoint_name = f"{MODEL_TYPE}_{MODEL_NR}_{fold}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = checkpoints_path,
        filename = checkpoint_name,
        save_top_k = SAVE_TOP_CKPTS,
        monitor = "val_C-index",
        mode = "max",
        enable_version_counter = True
    )

    # Logger:
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=f"{MODEL_TYPE}_{MODEL_NR}_fold_{fold}", job_type='train')
    wandb_logger.experiment.config.update({
        # Run Metadata
        "model_nr": MODEL_NR,
        "model_type": MODEL_TYPE,
        "local": LOCAL,
        "save_top_ckpts": SAVE_TOP_CKPTS,
        "wandb_project": WANDB_PROJECT,
        "do_cv": DO_CV,
        "full_training": FULL_TRAINING,
        # Data setup
        "modality": MODALITY,
        "bbox_or_centroid": BBOX_OR_CENTROID,
        # Training setup
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "tie_method": TIE_METHOD,
        "backbone_dropout": BCKB_DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "warmup_iters": WARMUP_ITERS,
        "num_workers": NUM_WORKERS
    })

    # Cleaning cache:
    torch.cuda.empty_cache()
    
    # Setting Trainer
    torch.set_float32_matmul_precision('medium')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator="gpu", devices=1, 
                         precision=32, max_epochs=EPOCHS,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=wandb_logger,
                         log_every_n_steps=24,
                         limit_val_batches=0.0 if fold=="all" else 1.0
                        )

    # Model setup
    model = HECKTOR_Model(
        model_type = MODEL_TYPE,
        load_pretrained = True,
        backbone_dropout = BCKB_DROPOUT,
        tie_method = TIE_METHOD,
        lr = LR,
        weight_decay = WEIGHT_DECAY,
        warmup_iters = WARMUP_ITERS,
        total_iters=total_steps
    )
    model.train()

    # Data Module setup
    dm = HECKTOR_DataModule(
        data_path = DATA_PATH,
        bbox_or_centroid = BBOX_OR_CENTROID,
        fold = fold,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        modality = MODALITY
    )

    trainer.fit(model, dm)

    # Free up memory
    del trainer
    del model
    del dm
    
    #Finishing run
    wandb.finish()


if DO_CV:
    for fold in range(1,6): # Iteration over folds
        train_model(fold=fold)
        
        
if FULL_TRAINING:
    train_model(fold="all")
