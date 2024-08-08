# IMPORT PACKAGES
from HECKTOR_Dataset import Mode, Modality
from HECKTOR_DataModule import HECKTOR_DataModule
from HECKTOR_Model import HECKTOR_Model, PCA_HECKTOR_Model

import math
import torch
import wandb
import pickle
from typing import Literal, Union, List, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from lifelines.utils import concordance_index

MODEL_NR:int = 6
FINETUNED_MODEL:str = "dino_vitb8_6"
MODEL_VERSION:Optional[Literal['v1', 'v2']] = None
N_COMPONENTS:int = 180 # Only specified to be logged on W&B.
LOCAL:bool = False
SAVE_TOP_CKPTS:int = 3
WANDB_PROJECT:str = "HECKTOR_PCA"
DO_CV:bool = True # Whether perform 5-Fold Cross-Validation.
FULL_TRAINING:bool = True # Whether train model on the whole training set, without CV. 
EPOCHS:int = 5
BATCH_SIZE:int = 16
MAX_LR:float = 5e-3
LR_ANNEAL_STRATEGY:Literal["linear", "cos"] = "cos"
INIT_DIV_FACTOR:int = 50
FINAL_DIV_FACTOR:int = 100
PCT_START:float = 0.1
TIE_METHOD:Literal["breslow", "efron"] = "breslow"
SELECTED_TRAIN_TRANSFORMS:List[Literal["elastic", "histogram"]] = ["elastic", "histogram"]
BCKB_DROPOUT:float = 0.12
NUM_WORKERS:int = 8

if LOCAL:
    DATA_PATH=""
    checkpoints_path=""
else:
    DATA_PATH="/home/dzban112/HECKTOR/Data/"
    checkpoints_path="/home/dzban112/HECKTOR/ckpt/"


def train_PCA_model(fold:Union[int, Literal["all"]] = 1):
    """
    ! Method designed to retrain only MLP HEAD of finetuned ViT utilizing PCA reduction of features between encoder and MLP HEAD. !
    
    If fold is integer, then model is trained on a specified CV fold and evaluated on a corresponding validational fold.
    When fold == "all", then model is trained using all training examples without evaluation on separate validational fold.
    """
    # Getting value of training steps:
    with open(DATA_PATH+f"train_data/train_fold_{fold}.pkl", "rb") as f:
        n_train_examples = len(pickle.load(f))
        steps_per_epoch = math.ceil(n_train_examples/BATCH_SIZE)
        total_steps = steps_per_epoch*EPOCHS

    # add a checkpoint callback that saves the model with the highest validation concordance index.
    checkpoint_name = f"PCA_{FINETUNED_MODEL}_{MODEL_NR}_{fold}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=checkpoint_name,
        save_top_k=1 if fold=="all" else SAVE_TOP_CKPTS,
        monitor=None if fold=="all" else "val_C-index",
        mode="max",
        enable_version_counter=True
    )

    # Logger:
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=f"{FINETUNED_MODEL}_{MODEL_NR}_fold_{fold}_PCA", job_type='train')
    wandb_logger.experiment.config.update({
        "model_nr": MODEL_NR,
        "finetuned model": FINETUNED_MODEL,
        "n_components": N_COMPONENTS,
        "local": LOCAL,
        "save_top_ckpts": SAVE_TOP_CKPTS,
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
                         log_every_n_steps=5,
                         limit_val_batches=0.0 if fold=="all" else 1.0
                        )
    
    model = PCA_HECKTOR_Model(
        ckpt_path=checkpoints_path+FINETUNED_MODEL+f"_{fold}{MODEL_VERSION if MODEL_VERSION is not None else ''}.ckpt",
        PCA_factors_path=DATA_PATH+f"train_data/PCA_factors/{FINETUNED_MODEL}_fold_{fold}.pt",
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
        modality=model.hparams.modality
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
        train_PCA_model(fold)
        
        
if FULL_TRAINING:
    train_PCA_model(fold="all")
    
