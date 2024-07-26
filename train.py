# IMPORT PACKAGES
from HECKTOR_Dataset import Mode, Modality
from HECKTOR_DataModule import HECKTOR_DataModule
from HECKTOR_Model import HECKTOR_Model

import math
import torch
import wandb
import pickle
from typing import Literal, Union, List
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from lifelines.utils import concordance_index


## HYPERPARAMETERS:
MODEL_NR:int = 25.2
LOCAL:bool = False
SAVE_TOP_CKPTS:int = 0
WANDB_PROJECT:str = "HECKTOR"
DO_CV:bool = True # Whether perform 5-Fold Cross-Validation.
FULL_TRAINING:bool = False # Whether train model on the whole training set, without CV. 
MODALITY:Modality = Modality.CT
MODEL_TYPE:Literal["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"] = "dino_vits8"
TRAINABLE_LAYERS:Union[int, Literal["all"]] = 0
EPOCHS:int = 25
BATCH_SIZE:int = 16
MAX_LR:float = 6e-5
LR_ANNEAL_STRATEGY:Literal["linear", "cos"] = "cos"
INIT_DIV_FACTOR:int = 100
FINAL_DIV_FACTOR:int = 100
PCT_START:float = 0.30
TIE_METHOD:Literal["breslow", "efron"] = "breslow"
SELECTED_TRAIN_TRANSFORMS:List[Literal["elastic", "crop", "contrast", "histogram"]] = ["elastic", "histogram", "crop"]
BCKB_DROPOUT:float = 0.12
NUM_WORKERS:int = 8

if MODALITY.value in ["CT", "PET"]: # IN_CHANS specifies if we use only CT/PET images in training or both.
    IN_CHANS:int = 1
elif MODALITY.value == "both":
    IN_CHANS:int = 2


if LOCAL:
    DATA_PATH=""
    checkpoints_path=""
else:
    DATA_PATH="/home/dzban112/HECKTOR/Data/"
    checkpoints_path="/home/dzban112/HECKTOR/ckpt/"

def train_model(fold:Union[int, Literal["all"]] = 1):
    """
    If fold is integer, then model is trained on a specified CV fold and evaluated on a corresponding validational fold.
    When fold == "all", then model is trained using all training examples without evaluation on separate validational fold.
    """
    # Getting value of training steps:
    with open(DATA_PATH+f"train_data/train_fold_{fold}.pkl", "rb") as f:
        n_train_examples = len(pickle.load(f))
        steps_per_epoch = math.ceil(n_train_examples/BATCH_SIZE)
        total_steps = steps_per_epoch*EPOCHS

    # add a checkpoint callback that saves the model with the lowest validation loss
    checkpoint_name = f"{MODEL_TYPE}_{MODEL_NR}_{fold}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=checkpoint_name,
        save_top_k=1 if fold=="all" else SAVE_TOP_CKPTS,
        monitor=None if fold=="all" else "val_C-index",
        mode="max",
        enable_version_counter=True
    )

    # Logger:
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=f"{MODEL_TYPE}_{MODEL_NR}_fold_{fold}", job_type='train')
    wandb_logger.experiment.config.update({
        "model_nr": MODEL_NR,
        "local": LOCAL,
        "save_top_ckpts": SAVE_TOP_CKPTS,
        "modality": MODALITY,
        "model_type": MODEL_TYPE,
        "in_chans": IN_CHANS,
        "trainable_layers": TRAINABLE_LAYERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_lr": MAX_LR,
        "lr_anneal_strategy": LR_ANNEAL_STRATEGY,
        "init_div_factor": INIT_DIV_FACTOR,
        "final_div_factor": FINAL_DIV_FACTOR,
        "pct_start": PCT_START,
        "tie_method": TIE_METHOD,
        "selected_train_transforms": SELECTED_TRAIN_TRANSFORMS,
        "backbone_dropout": BCKB_DROPOUT,
        "num_workers": NUM_WORKERS
    })

    # Cleaning cache:
    torch.cuda.empty_cache()
    
    torch.set_float32_matmul_precision('medium')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator="gpu", devices=1, 
                         precision=32, max_epochs=EPOCHS,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=wandb_logger,
                         log_every_n_steps=20,
                         limit_val_batches=0.0 if fold=="all" else 1.0
                        )

    model = HECKTOR_Model(
        model_type=MODEL_TYPE,
        in_chans=IN_CHANS,
        trainable_layers=TRAINABLE_LAYERS,
        backbone_dropout=BCKB_DROPOUT,
        max_lr=MAX_LR,
        tie_method=TIE_METHOD,
        lr_anneal_strategy=LR_ANNEAL_STRATEGY,
        init_div_factor=INIT_DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
        total_steps=total_steps,
        pct_start=PCT_START
    )

    dm = HECKTOR_DataModule(
        fold=fold,
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        selected_train_transforms=SELECTED_TRAIN_TRANSFORMS,
        modality=MODALITY
    )

    trainer.fit(model, dm)

    # Free up memory
    del trainer
    del model
    del dm
    
    #Finishing run
    wandb.finish()


if DO_CV:
    for fold in range(1,2): # Iteration over folds
        train_model(fold=fold)
        
if FULL_TRAINING:
    train_model(fold="all")
    
