from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from lassonet1.lassonet.cox import CoxPHLoss


def set_encoder_dropout_p(module, dropout_p):
    if isinstance(module, nn.TransformerEncoderLayer):
        # Traverse the encoder layer to find dropout layers
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Dropout):
                # Sets dropout probability for dropout layers within encoder blocks
                child_module.p = dropout_p


class TwoHeadsBeast(LightningModule):
    __doc__ = """Powerfull Beast"""
    def __init__(
        self,
        learning_rate=1e-3,
        tie_method="breslow",
        lr_schedule="fixed",
        milestones=[5,10],
        patience=5,
        trainable_layers=0,
        dropout=0.0
    ):
        super().__init__()
        self.save_hyperparameters()
        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.lr_schedule = lr_schedule
        self.milestones = milestones
        self.patience = patience
        self.criterion = CoxPHLoss(tie_method)
        self.CT_path = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        self.PET_path = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        # changing dropout values in the encoder blocks:
        if dropout > 0.0:       
            self.CT_path.apply(lambda module: set_encoder_dropout_p(module, dropout_p=self.dropout))
            self.PET_path.apply(lambda module: set_encoder_dropout_p(module, dropout_p=self.dropout))
        CT_all_layers = len(list(self.CT_path.parameters()))
        PET_all_layers = len(list(self.PET_path.parameters()))
        for i, p1, p2 in enumerate(zip(self.CT_path.parameters(), self.PET_path.parameters())):
            if i < (len(CT_all_layers) - trainable_layers):
                p1.requires_grad = False
                p2.requires_grad = False
        self.l1 = nn.Linear(768, 384)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(384, 1)
        

    def forward(self, CT, PET):
        x1 = self.CT_path(CT)
        x2 = self.PET_path(PET)
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

    def training_step(self, batch, batch_idx):
        CT_crop = batch["CT"]
        PET_crop = batch["PET"]
        labels = batch["labels"]
        # Check if all events are censored, if so, skip this batch
        # CoxPHLoss is not well defined in that case
        if labels[:,1].sum() == 0:
            return None
        logits = self(CT_crop, PET_crop)
        loss = self.criterion(logits, labels)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        CT_crop = batch["CT"]
        PET_crop = batch["PET"]
        labels = batch["labels"]
        # Check if all events are censored, if so, skip this batch
        if labels[:,1].sum() == 0:
            return None
        logits = self(CT_crop, PET_crop)
        loss = self.criterion(logits, labels)
        self.log(
            "val_loss", loss, prog_bar=True, sync_dist=False
        )
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        if self.lr_schedule == "fixed":
            scheduler = MultiStepLR(optimizer, self.milestones)
        elif self.lr_schedule == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, patience=self.patience)
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "name": "learning_rate",
        }
        return [optimizer], [lr_scheduler]
