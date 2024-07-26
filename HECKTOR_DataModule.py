import pytorch_lightning as pl
from HECKTOR_Dataset import HECKTOR_Dataset, Mode, Modality
from typing import List, Literal, Union
import pickle
from torch.utils.data import DataLoader
from monai import transforms as T
import os

CWD = os.getcwd()

class HECKTOR_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path:str=CWD+"/Data/",
                 fold:Union[int, Literal["all"]] = 1,
                 batch_size:int=32,
                 num_workers:int=8,
                 selected_train_transforms:List[Literal["elastic", "crop", "contrast", "histogram"]]=["elastic", "crop"],
                 modality:Modality = Modality.CT
                ):
        super().__init__()
        self.fold = fold
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.selected_train_transforms = selected_train_transforms
        self.modality = modality
        self.has_val = fold != "all"

    
    def setup(self, stage=None):
        # Defining MONAI transforms. (Normalization is done separately in the Dataset module.)
        transform_dict = {
            "elastic": T.Rand3DElastic(
                sigma_range=[1, 1.5],
                magnitude_range=[1, 3],
                prob=0.3,
                rotate_range=[0.1, 0.3],
                shear_range=[0.1, 0.3],
                translate_range=[0.1, 0.3],
                scale_range=[0.1, 0.3],
                padding_mode="border",
            ),
            "crop": T.RandSpatialCrop(
                roi_size=[90, 90], random_center=True, random_size=True
            ),
            "contrast": T.RandAdjustContrast(prob=0.3, gamma=(0.5, 1.5)),
            "histogram": T.RandHistogramShift(prob=0.3, num_control_points=8),
        }
        
        train_transforms = T.Compose(
            [transform_dict[transform] for transform in self.selected_train_transforms]
        )

        # Transforms for validation data and test data are the same.
        if "crop" in self.selected_train_transforms:
            test_transforms = T.Compose([T.CenterSpatialCrop(roi_size=[90, 90])])
        else:
            test_transforms = T.Compose([])
            
        self.train_ds = HECKTOR_Dataset(
            data_path=self.data_path,
            fold=self.fold,
            transform=train_transforms,
            mode=Mode.TRAIN,
            modality=self.modality
        )

        if self.has_val:
            self.val_ds = HECKTOR_Dataset(
                data_path=self.data_path,
                fold=self.fold,
                transform=test_transforms,
                mode=Mode.VAL,
                modality=self.modality
            )

        self.test_ds = HECKTOR_Dataset(
            data_path=self.data_path,
            fold=self.fold,
            transform=test_transforms,
            mode=Mode.TEST,
            modality=self.modality
        )


    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers
                                 )
        return train_loader

    
    def val_dataloader(self):
        if self.has_val:
            val_loader = DataLoader(self.val_ds,
                                    shuffle=False,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers
                                   )
            return val_loader
        return DataLoader([], batch_size=self.batch_size, num_workers=self.num_workers) # Empty DataLoader.

    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds,
                                shuffle=False,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers
                               )
        return test_loader
