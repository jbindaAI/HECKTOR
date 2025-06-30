from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from typing import Literal, List
import torch
from lifelines.utils import concordance_index
from lightning_modules.models import VanillaViT, DeiT, DINO, DINOv2, VisionCLIP

from loss_fn.loss_fn import CoxPHLoss


def load_backbone(model_type: Literal["vanilla_vit", "deit_vit", "dino_vit", "dino_v2_reg", "clip"],
                  load_pretrained: bool = False,
                  backbone_dropout: float = 0.12
                  ):
    if model_type == "vanilla_vit":
        backbone = VanillaViT(head_dim = 1,
                              backbone_dropout = backbone_dropout,
                              load_pretrained_backbone = load_pretrained
                              )
    elif model_type == "deit_vit":
        backbone = DeiT(model_size="base",
                        head_dim = 1,
                        backbone_dropout = backbone_dropout,
                        load_pretrained_backbone = load_pretrained
                        )
    elif model_type == "dino_vit":
        backbone = DINO(model_size="small",
                        patch_size=8,
                        head_dim = 1,
                        backbone_dropout = backbone_dropout,
                        load_pretrained_backbone = load_pretrained
                        )
    elif model_type == "dino_v2_reg":
        backbone = DINOv2(model_size="base",
                          head_dim = 1,
                          backbone_dropout = backbone_dropout,
                          load_pretrained_backbone = load_pretrained
                          )
    elif model_type == "clip":
        backbone = VisionCLIP(head_dim = 1,
                          backbone_dropout = backbone_dropout,
                          load_pretrained_backbone = load_pretrained
                          )
    else:
        raise ValueError("Expected model type to be one of: vanilla_vit, deit_vit, dino_vit, dino_v2_reg, clip,"
                         f" but got {model_type} instead.")
    return backbone


class HECKTOR_Model(LightningModule):
    __doc__ = """End2End model which takes both CT and PET"""
    def __init__(
        self,
        model_type: Literal["vanilla_vit", "deit_vit", "dino_vit", "dino_v2_reg", "clip"],
        load_pretrained: bool = False,
        backbone_dropout: float = 0.0,
        tie_method: Literal["breslow", "efron"] = "breslow",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_iters: int = 100,
        total_iters: int = 1000
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.load_pretrained = load_pretrained
        self.backbone_dropout = backbone_dropout
        self.tie_method = tie_method
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        
        # Criterion Setup
        self.criterion = CoxPHLoss(tie_method)

        # Backbone Setup
        self.backbone = load_backbone(model_type = self.model_type,
                                      load_pretrained = self.load_pretrained,
                                      backbone_dropout = self.backbone_dropout)
        
        # Accumulating results to compute C-index at the epoch end.
        self.test_preds = []
        self.test_labels = []
        self.val_preds = []
        self.val_labels = []
        
        
    def forward(self, imgs: torch.Tensor, only_embedding: bool = False):
        if only_embedding:
            return self.backbone._get_embedding(imgs)
        else:
            return self.backbone(imgs)


    # def common_eval_step(self, batch: torch.Tensor) -> List[torch.Tensor, torch.Tensor]:
    #     crop_list = batch["crop"]
    #     labels = batch["labels"]
    #     slice_logits = []
    #     for crop in crop_list:
    #         logits = self.forward(crop)
    #         slice_logits.append(logits)
    #     logits = torch.stack(slice_logits)
    #     logits = torch.mean(logits, dim=0)
    #     # Check if all events are censored, if so, skip this batch
    #     # CoxPHLoss is not well defined in that case
    #     if labels[:,1].sum() == 0:
    #         return [logits]
    #     loss = self.criterion(logits, labels)
    #     return [logits, loss]


    def common_step(self, batch: torch.Tensor) -> List[torch.Tensor]:
        crops = batch["crops"]
        labels = batch["labels"]
        logits = self.forward(crops)
        # Check if all events are censored, if so, skip this batch
        # CoxPHLoss is not well defined in that case
        if labels[:,1].sum() == 0:
            return [logits]
        loss = self.criterion(logits, labels)
        return [logits, loss]


    def compute_c_index(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        rfs = labels[:, 0]
        relapse = labels[:, 1]
        # The model is trained to predict the hazard (large values lead to early relapse) 
        # so we need to flip the sign of predictions because the concordance index is 
        # defined for the survival time (smaller values mean early relapse)
        c_index = concordance_index(rfs, -preds, relapse)
        return c_index


    def training_step(self, batch):
        results = self.common_step(batch)
        if len(results) == 1:
            return None
        else:
            loss = results[1]
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss


    def validation_step(self, batch):
        results = self.common_step(batch)
        logits = results[0]
        # Accumulating logits and labels for c-index computation at the end of validation.
        self.val_preds.append(logits.detach().cpu())
        self.val_labels.append(batch["labels"].cpu())
        if len(results) == 1:
            return None
        else:
            loss = results[1]
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss


    def on_validation_epoch_end(self):
        # Concatenates all predictions and labels
        val_preds = torch.cat(self.val_preds).squeeze().numpy()
        val_labels = torch.cat(self.val_labels).numpy()
        c_index = self.compute_c_index(preds=val_preds, labels=val_labels)
        self.log("val_C-index", c_index)
        self.val_preds.clear()
        self.val_labels.clear()


    def test_step(self, batch):
        results = self.common_step(batch)
        logits = results[0]
        # Accumulating logits and labels for c-index computation at the end of testing.
        self.test_preds.append(logits.detach().cpu())
        self.test_labels.append(batch["labels"].cpu())
        if len(results) == 1:
            return None
        else:
            loss = results[1]
            self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss


    def on_test_epoch_end(self):
        # Concatenates all predictions and labels
        test_preds = torch.cat(self.test_preds).squeeze().numpy()
        test_labels = torch.cat(self.test_labels).numpy()
        c_index = self.compute_c_index(preds=test_preds, labels=test_labels)
        self.log("test_C-index", c_index)
        self.test_preds.clear()
        self.test_labels.clear()


    @staticmethod
    def _make_scheduler(optimizer: torch.optim.Optimizer, warmup_iters: int, total_iters: int) -> LambdaLR:
        warmup_iters = min(warmup_iters, total_iters)

        def lr_lambda(current_iter: int) -> float:
            if current_iter < warmup_iters:
                return current_iter / warmup_iters
            return max(0.1, (total_iters - current_iter) / (total_iters - warmup_iters))

        return LambdaLR(optimizer, lr_lambda)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.backbone.parameters(), 
                                      lr = self.lr, 
                                      weight_decay = self.weight_decay)
        lr_scheduler = self._make_scheduler(optimizer, self.warmup_iters, self.total_iters)

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler]


# class PCA_HECKTOR_Model(HECKTOR_Model):
#     def __init__(
#         self,
#         ckpt_path: str,
#         PCA_factors_path: str,
#         backbone_dropout:float=0.0,
#         max_lr:float=1e-3,
#         tie_method:Literal["breslow", "efron"]="breslow",
#         lr_anneal_strategy:Literal["linear", "cos"]="cos",
#         init_div_factor:int=100,
#         final_div_factor:int=10000,
#         total_steps:int=100,
#         pct_start:float=0.10
#     ):
#         # Load the already finetuned model from the checkpoint
#         model = HECKTOR_Model.load_from_checkpoint(ckpt_path)
#         # Initialize the parent class (HECKTOR_Model)
#         super().__init__(
#             model_type=model.hparams.model_type,
#             in_chans=model.hparams.in_chans,
#             modality=model.hparams.modality,
#             trainable_layers=0,
#             backbone_dropout=backbone_dropout,
#             max_lr=max_lr,
#             tie_method=tie_method,
#             lr_anneal_strategy=lr_anneal_strategy,
#             init_div_factor=init_div_factor,
#             final_div_factor=final_div_factor,
#             total_steps=total_steps,
#             pct_start=pct_start
#         )

#         # Taking finetuned backbone
#         self.backbone = model.backbone

#         # Load PCA components and mean
#         pca_data = torch.load(PCA_factors_path)
#         self.pca_components = pca_data['components']
#         self.pca_mean = pca_data['mean']

#         # Freeze the backbone parameters of the already finetuned model
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#         # Replace the MLP head with a new one that accepts reduced PCA features
#         self.mlp_head = nn.Sequential(nn.Linear(self.pca_components.shape[0], 1))

#     def forward(self, imgs):
#         # Forward pass through the backbone
#         x = self.backbone(imgs)

#         # Apply PCA transformation
#         x = x - self.pca_mean.to(self.device)
#         x = torch.matmul(x, self.pca_components.T.to(self.device))

#         # Forward pass through the new MLP head
#         x = self.mlp_head(x)
#         return x


