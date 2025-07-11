from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from lifelines.utils import concordance_index
from torch import nn
import torch
from typing import Optional, Literal

from hecktor_survival.loss_fn.loss_fn import CoxPHLoss
from vit_zoo import build_model


class BaseHECKTORModel(LightningModule):
    def __init__(
        self,
        model_type: str,
        load_pretrained: bool = False,
        backbone_dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_iters: int = 100,
        total_iters: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = build_model(
            model_type=model_type,
            backbone_dropout=backbone_dropout,
            load_pretrained_backbone=load_pretrained,
        )
        self.embedding_dim = self.backbone.get_output_dim()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": self._make_scheduler(
                optimizer, self.hparams.warmup_iters, self.hparams.total_iters
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def _make_scheduler(optimizer, warmup_iters, total_iters):
        def lr_lambda(current_step):
            if current_step < warmup_iters:
                return float(current_step) / float(max(1, warmup_iters))
            return max(0.1, (total_iters - current_step) / max(1, total_iters - warmup_iters))

        return LambdaLR(optimizer, lr_lambda)


class VanillaSurvivalModel(BaseHECKTORModel):
    def __init__(self, tie_method: Literal["efron", "breslow"] = "breslow", **kwargs):
        super().__init__(**kwargs)
        self.rfs_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.surv_loss = CoxPHLoss(tie_method)
        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.rfs_head(features).squeeze(-1)

    def training_step(self, batch: dict) -> torch.Tensor:
        rfs_pred = self.forward(batch["crops"])
        # Check if all events are censored, if so, skip this batch
        # CoxPHLoss is not well defined in that case
        if batch["labels"][:,1].sum() == 0:
            print("Skipping censored!!")
            return
        else:
            loss = self.surv_loss(rfs_pred.squeeze(), batch["labels"])
            self.log("train_loss", loss)
            return loss

    def validation_step(self, batch: dict) -> torch.Tensor:
        rfs_pred = self.forward(batch["crops"])
        # Accumulating logits and labels for c-index computation at the end of validation.
        self.val_preds.append(rfs_pred.detach().cpu())
        self.val_labels.append(batch["labels"].cpu())
        
        if batch["labels"][:,1].sum() == 0:
            return
        else:
            loss = self.surv_loss(rfs_pred.squeeze(), batch["labels"])
            self.log("val_loss", loss)
            return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).squeeze().numpy()
        labels = torch.cat(self.val_labels).numpy()
        c_index = concordance_index(labels[:, 0], -preds, labels[:, 1])
        self.log("val_C-index", c_index)
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch: dict) -> torch.Tensor:
        rfs_pred = self.forward(batch["crops"])
        # Accumulating logits and labels for c-index computation at the end of test.
        self.test_preds.append(rfs_pred.detach().cpu())
        self.test_labels.append(batch["labels"].cpu())

        if batch["labels"][:,1].sum() == 0:
            return
        else:
            loss = self.surv_loss(rfs_pred.squeeze(), batch["labels"])
            self.log("test_loss", loss)
            return loss

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).squeeze().numpy()
        labels = torch.cat(self.test_labels).numpy()
        c_index = concordance_index(labels[:, 0], -preds, labels[:, 1])
        self.log("test_C-index", c_index)
        self.test_preds.clear()
        self.test_labels.clear()


class SimplifiedAreaModel(BaseHECKTORModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.area_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.reg_loss = nn.HuberLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.area_head(features).squeeze(-1)

    def training_step(self, batch: dict) -> torch.Tensor:
        area_pred = self.forward(batch["crops"])
        loss = self.reg_loss(area_pred, batch["cancer_area"].float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict) -> torch.Tensor:
        area_pred = self.forward(batch["crops"])
        loss = self.reg_loss(area_pred, batch["cancer_area"].float())
        self.log("val_loss", loss)
        return loss


class AdvancedHybridModel(BaseHECKTORModel):
    def __init__(self, aux_loss_weight: float = 0.3, tie_method: Literal["efron", "breslow"] = "breslow", **kwargs):
        super().__init__(**kwargs)
        self.aux_loss_weight = aux_loss_weight

        self.rfs_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.area_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.surv_loss = CoxPHLoss(tie_method)
        self.reg_loss = nn.HuberLoss()

        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        rfs_pred = self.rfs_head(features).squeeze(-1)
        area_pred = self.area_head(features).squeeze(-1)
        return rfs_pred, area_pred

    def training_step(self, batch: dict) -> torch.Tensor:
        rfs_pred, area_pred = self.forward(batch["crops"])
        if batch["labels"][:,1].sum() == 0:
            print("Skipping censored!!")
            return None
        else:
            loss1 = self.surv_loss(rfs_pred, batch["labels"])
            loss2 = self.reg_loss(area_pred, batch["cancer_area"].float())
            loss = (1 - self.aux_loss_weight) * loss1 + self.aux_loss_weight * loss2
            self.log_dict({"train_surv_loss": loss1, "train_reg_loss": loss2, "train_loss": loss})
            return loss

    def validation_step(self, batch: dict) -> torch.Tensor:
        rfs_pred, area_pred = self.forward(batch["crops"])
        if batch["labels"][:,1].sum() == 0:
            print("Skipping censored!!")
            return None
        else:
            loss1 = self.surv_loss(rfs_pred, batch["labels"])
            loss2 = self.reg_loss(area_pred, batch["cancer_area"].float())
            loss = (1 - self.aux_loss_weight) * loss1 + self.aux_loss_weight * loss2
            self.val_preds.append(rfs_pred.detach().cpu())
            self.val_labels.append(batch["labels"].cpu())
            self.log_dict({"val_surv_loss": loss1, "val_reg_loss": loss2, "val_loss": loss})
            return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).squeeze().numpy()
        labels = torch.cat(self.val_labels).numpy()
        c_index = concordance_index(labels[:, 0], -preds, labels[:, 1])
        self.log("val_C-index", c_index)
        self.val_preds.clear()
        self.val_labels.clear()
    
    def test_step(self, batch: dict) -> torch.Tensor:
        rfs_pred = self.forward(batch["crops"])
        if batch["labels"][:,1].sum() == 0:
            print("Skipping censored!!")
            return None
        else:
            loss = self.surv_loss(rfs_pred, batch["labels"])
            self.test_preds.append(rfs_pred.detach().cpu())
            self.test_labels.append(batch["labels"].cpu())
            self.log("test_loss", loss)
            return loss

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).squeeze().numpy()
        labels = torch.cat(self.test_labels).numpy()
        c_index = concordance_index(labels[:, 0], -preds, labels[:, 1])
        self.log("test_C-index", c_index)
        self.test_preds.clear()
        self.test_labels.clear()
