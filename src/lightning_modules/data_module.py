import pytorch_lightning as pl
from typing import Literal, Union, Optional
from torch.utils.data import DataLoader, Dataset
import monai.transforms as mt

from lightning_modules.dataset import HECKTOR_Dataset


class HECKTOR_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        bbox_or_centroid: Literal["bbox", "centroid"],
        fold: Union[int, Literal["all"]] = 1,
        batch_size: int = 32,
        num_workers: int = 8,
        modality: Literal["CT", "PET", "Merged"] = "CT"
    ):
        """
        PyTorch Lightning DataModule for the HECKTOR dataset.

        Args:
            data_path: Path to the dataset root.
            bbox_or_centroid: Use bounding box or centroid crops.
            fold: Fold number or "all" for cross-validation.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for DataLoaders.
            selected_train_transforms: List of transforms to apply during training.
            modality: Imaging modality ("CT", "PET", "Merged").
        """
        super().__init__()
        self.fold = fold
        self.data_path = data_path
        self.bbox_or_centroid = bbox_or_centroid
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modality = modality
        self.has_val = fold != "all"
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up datasets for different stages.
        """
        if stage == "fit" or stage is None:
            train_3D_transforms = mt.Compose([
                mt.RandFlipd(keys=["CT", "PET"], prob=0.5, spatial_axis=2, allow_missing_keys=True),
                mt.RandAffined(keys=["CT", "PET"], prob=0.7, rotate_range=(0.1,0.1,0.1), allow_missing_keys=True),
                mt.RandGaussianNoised(keys=["PET"], prob=0.2, allow_missing_keys=True)
                ])

            # Train DS
            self.train_ds = HECKTOR_Dataset(
                data_path=self.data_path,
                bbox_or_centroid=self.bbox_or_centroid,
                monai_3D_transform=train_3D_transforms,
                fold=self.fold,
                mode="train",
                modality=self.modality
            )
            # Optional Val DS
            if self.has_val:
                self.val_ds = HECKTOR_Dataset(
                    data_path=self.data_path,
                    bbox_or_centroid=self.bbox_or_centroid,
                    monai_3D_transform=None,
                    fold=self.fold,
                    mode="val",
                    modality=self.modality
                )

        if stage == "test" or stage is None:
            # Test DS
            self.test_ds = HECKTOR_Dataset(
                data_path=self.data_path,
                bbox_or_centroid=self.bbox_or_centroid,
                monai_3D_transform=None,
                fold=self.fold,
                mode="test",
                modality=self.modality
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for training set.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for validation set.
        """
        if self.has_val and self.val_ds is not None:
            return DataLoader(
                self.val_ds,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        return DataLoader([], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for test set.
        """
        return DataLoader(
            self.test_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )