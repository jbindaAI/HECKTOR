# IMPORT PACKAGES
from hecktor_survival.lightning_modules.data_module import HECKTOR_DataModule
from hecktor_survival.lightning_modules.model_module import VanillaSurvivalModel, SimplifiedAreaModel, AdvancedHybridModel

import os
import math
import torch
import wandb
import pickle
from typing import Literal, Type
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# HYPERPARAMETERS:
## Run Metadata:
MODEL_NR: int = 5
## General setup
MODEL_TYPE: Literal["vanilla_vit", "deit_vit", "dino_vit", "dino_v2", "clip"] = "clip"
MODEL_CLASS: Type[pl.LightningModule] = AdvancedHybridModel
LOCAL: bool = True
SAVE_TOP_CKPTS: int = 1
WANDB_PROJECT: str = "HECKTOR_advanced"
## Data setup
MODALITY: Literal["CT", "PET", "Merged"] = "Merged"
BBOX_OR_CENTROID: Literal["bbox", "centroid"] = "bbox"
## Training setup
EPOCHS: int = 30
BATCH_SIZE: int = 16
LR: float = 3e-4
TIE_METHOD: Literal["breslow", "efron"] = "breslow"
BCKB_DROPOUT: float = 0.12
WEIGHT_DECAY: float = 0.01
WARMUP_ITERS: int = 100
NUM_WORKERS: int = 8
AUX_LOSS_WEIGHT: float = 0.25  # only used for AdvancedHybridModel

# Paths
if LOCAL:
    DATA_PATH = "/home/jakub/research/HECKTOR/Data"
    checkpoints_path = "/home/jakub/research/HECKTOR/ckpt"
else:
    DATA_PATH = "/home/dzban112/HECKTOR/Data/"
    checkpoints_path = "/home/dzban112/HECKTOR/ckpt/"

def train_model(fold: int = 1):
    with open(os.path.join(DATA_PATH, "filtered_labels", f"train_labels/train_fold_{fold}.pkl"), "rb") as f:
        n_train_examples = len(pickle.load(f))
        steps_per_epoch = math.ceil(n_train_examples / BATCH_SIZE)
        total_steps = steps_per_epoch * EPOCHS

    checkpoint_name = f"{MODEL_TYPE}_{MODEL_CLASS.__name__}_{MODEL_NR}_{fold}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=checkpoint_name,
        save_top_k=SAVE_TOP_CKPTS,
        monitor="val_C-index",
        mode="max",
        enable_version_counter=True,
    )

    # Logger
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=f"{MODEL_TYPE}_{MODEL_CLASS.__name__}_{MODEL_NR}_fold_{fold}",
        job_type="train"
    )
    wandb_logger.experiment.config.update({
        "model_nr": MODEL_NR,
        "model_type": MODEL_TYPE,
        "model_class": MODEL_CLASS.__name__,
        "local": LOCAL,
        "save_top_ckpts": SAVE_TOP_CKPTS,
        "wandb_project": WANDB_PROJECT,
        "modality": MODALITY,
        "bbox_or_centroid": BBOX_OR_CENTROID,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "tie_method": TIE_METHOD,
        "backbone_dropout": BCKB_DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "warmup_iters": WARMUP_ITERS,
        "num_workers": NUM_WORKERS,
        "aux_loss_weight": AUX_LOSS_WEIGHT
    })

    # Clean cache
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        logger=wandb_logger,
        log_every_n_steps=24
    )

    # Instantiate model
    if MODEL_CLASS is VanillaSurvivalModel:
        model = MODEL_CLASS(
            model_type=MODEL_TYPE,
            load_pretrained=True,
            backbone_dropout=BCKB_DROPOUT,
            tie_method=TIE_METHOD,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_iters=WARMUP_ITERS,
            total_iters=total_steps
        )
    elif MODEL_CLASS is SimplifiedAreaModel:
        model = MODEL_CLASS(
            model_type=MODEL_TYPE,
            load_pretrained=True,
            backbone_dropout=BCKB_DROPOUT,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_iters=WARMUP_ITERS,
            total_iters=total_steps
        )
    elif MODEL_CLASS is AdvancedHybridModel:
        model = MODEL_CLASS(
            model_type=MODEL_TYPE,
            load_pretrained=True,
            backbone_dropout=BCKB_DROPOUT,
            tie_method=TIE_METHOD,
            aux_loss_weight=AUX_LOSS_WEIGHT,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_iters=WARMUP_ITERS,
            total_iters=total_steps
        )
    else:
        raise ValueError("Unsupported model class!")

    model.train()

    # DataModule
    dm = HECKTOR_DataModule(
        data_path=DATA_PATH,
        bbox_or_centroid=BBOX_OR_CENTROID,
        fold=fold,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        modality=MODALITY
    )

    trainer.fit(model, dm)

    del trainer, model, dm
    wandb.finish()


for fold in range(1, 2):
    train_model(fold=fold)