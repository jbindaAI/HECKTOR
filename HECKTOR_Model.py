from pytorch_lightning import LightningModule
from typing import Literal, Union
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from lassonet1.lassonet.cox import CoxPHLoss
from DINO_ViT_source.vision_transformer import instantiate_model, weights_map
from lifelines.utils import concordance_index


def set_encoder_dropout_p(module, dropout_p):
    if isinstance(module, nn.Dropout):
        # Sets dropout probability for dropout layers within encoder blocks.
        module.p = dropout_p


class HECKTOR_Model(LightningModule):
    __doc__ = """End2End model which takes both CT and PET"""
    def __init__(
        self,
        model_type:Literal["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"],
        in_chans:int=1,
        trainable_layers:Union[int, Literal["all"]]="all",
        backbone_dropout:float=0.0,
        max_lr:float=1e-3,
        tie_method:Literal["breslow", "efron"]="breslow",
        lr_anneal_strategy:Literal["linear", "cos"]="cos",
        init_div_factor:int=100,
        final_div_factor:int=10000,
        total_steps:int=100,
        pct_start:float=0.10
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_type=model_type
        self.in_chans=in_chans
        self.trainable_layers=trainable_layers
        self.backbone_dropout=backbone_dropout
        self.max_lr=max_lr
        self.tie_method=tie_method
        self.lr_anneal_strategy=lr_anneal_strategy
        self.init_div_factor=init_div_factor
        self.final_div_factor=final_div_factor
        self.total_steps=total_steps
        self.pct_start=pct_start

        self.criterion = CoxPHLoss(tie_method)

        ## Backbone setting:
        if model_type in ["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"]:
            state_dict = torch.load(f"DINO_ViT_source/pretrained/{weights_map[model_type]}")
            # Adjusting embedding weights with "weight inflation" strategy:
            emb_weights = state_dict["patch_embed.proj.weight"]
            emb_weights = emb_weights.sum(1, keepdim=True) # Reducing color channels to one channel - gray channel.
            emb_weights = emb_weights.repeat(1, self.in_chans, 1, 1)/self.in_chans # Inflating weights.
            state_dict["patch_embed.proj.weight"] = emb_weights
            # Instantiating model:
            self.backbone = instantiate_model(model_type, in_chans=self.in_chans)
            self.backbone.load_state_dict(state_dict, strict=True)
            self.mlp_head=nn.Sequential(nn.Linear(self.backbone.embed_dim, 1))
        else:
            raise Exception("Provided model name is not recognized.")
        
        # changing dropout values in the backbone:
        if backbone_dropout > 0.0:       
            self.backbone.apply(lambda module: set_encoder_dropout_p(module, dropout_p=self.backbone_dropout))
        
        if trainable_layers != "all":
            all_layers = len(list(self.backbone.parameters()))
            for i, p in enumerate(self.backbone.parameters()):
                if i < (all_layers - trainable_layers):
                    p.requires_grad = False
        
        # Accumulating results to compute C-index at the epoch end.
        self.test_preds = []
        self.test_labels = []
        self.val_preds = []
        self.val_labels = []
        
    def forward(self, imgs):
        x = self.backbone(imgs)
        x = self.mlp_head(x)
        return x

    def common_step(self, batch):
        crops = batch["crop"]
        labels = batch["labels"]
        logits = self.forward(crops)
        # Check if all events are censored, if so, skip this batch
        # CoxPHLoss is not well defined in that case
        if labels[:,1].sum() == 0:
            return [logits]
        loss = self.criterion(logits, labels)
        return [logits, loss]

    def compute_c_index(self, preds, labels):
        rfs = labels[:, 0]
        relapse = labels[:, 1]
        # The model is trained to predict the hazard (large values lead to early relapse) 
        # so we need to flip the sign of predictions because the concordance index is 
        # defined for the survival time (smaller values mean early relapse)
        c_index = concordance_index(rfs, -preds, relapse)
        return c_index

    def training_step(self, batch, batch_idx):
        results = self.common_step(batch)
        if len(results) == 1:
            return None
        else:
            loss = results[1]
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

    def validation_step(self, batch, batch_idx):
        results = self.common_step(batch)
        logits = results[0]
        # Accumulating logits and labels for c-index computation at the end of validation.
        self.val_preds.append(logits)
        self.val_labels.append(batch["labels"])
        if len(results) == 1:
            return None
        else:
            loss = results[1]
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

    def on_validation_epoch_end(self):
        # Concatenates all predictions and labels
        val_preds = torch.cat(self.val_preds).squeeze().cpu().numpy()
        val_labels = torch.cat(self.val_labels).cpu().numpy()
        c_index = self.compute_c_index(preds=val_preds, labels=val_labels)
        self.log("val_C-index", c_index)
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        results = self.common_step(batch)
        logits = results[0]
        # Accumulating logits and labels for c-index computation at the end of testing.
        self.test_preds.append(logits)
        self.test_labels.append(batch["labels"])
        if len(results) == 1:
            return None
        else:
            loss = results[1]
            self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss
            
    def on_test_epoch_end(self):
        # Concatenates all predictions and labels
        test_preds = torch.cat(self.test_preds).squeeze().cpu().numpy()
        test_labels = torch.cat(self.test_labels).cpu().numpy()
        c_index = self.compute_c_index(preds=test_preds, labels=test_labels)
        self.log("test_C-index", c_index)
        self.test_preds.clear()
        self.test_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           div_factor=self.init_div_factor,
                                                           final_div_factor=self.final_div_factor,
                                                           max_lr=self.max_lr,
                                                           cycle_momentum=True,
                                                           pct_start=self.pct_start,
                                                           anneal_strategy=self.lr_anneal_strategy,
                                                           total_steps=self.total_steps
                                                          )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler]
