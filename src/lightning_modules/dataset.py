import pandas as pd
from torch.utils.data import Dataset
import torch
import torchvision.transforms.v2.functional as TF
import numpy as np
import random
import os
from typing import Optional, Literal, Union, Callable, Dict, Any
import pickle

class HECKTOR_Dataset(Dataset):
    """
    PyTorch Dataset for HECKTOR slices with RFS/Relapse labels.
    """

    def __init__(
        self,
        data_path: str,
        bbox_or_centroid: Literal["bbox", "centroid"],
        monai_3D_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fold: Union[int, Literal["all"]] = 1,
        mode: Literal["train", "val", "test"] = "train",
        modality: Literal["CT", "PET", "Merged"] = "CT"
    ):
        """
        Args:
            data_path: Root directory of the dataset.
            bbox_or_centroid: Whether to use bounding box or centroid crops.
            monai_3D_transform: Optional MONAI 3D transforms to apply to crops.
            fold: Fold number or "all" for cross-validation.
            mode: Dataset split ("train", "val", "test").
            modality: Imaging modality ("CT", "PET", "Merged").
        """
        self.data_path = data_path
        self.bbox_or_centroid = bbox_or_centroid
        self.monai_3D_transform = monai_3D_transform
        self.fold = fold
        self.mode = mode
        self.modality = modality

        # Patient IDs and Labels
        if self.mode == "test":
            labels_path = os.path.join(self.data_path, "filtered_labels/test_labels/test_labels_filtered.csv")
            self.labels = pd.read_csv(labels_path).set_index("PatientID")[["RFS", "Relapse"]]
            self.pids = list(self.labels.index)
        else:  # Train or Val
            labels_path = os.path.join(self.data_path, "filtered_labels/train_labels/train_labels_filtered.csv")
            labels = pd.read_csv(labels_path).set_index("PatientID")[["RFS", "Relapse"]]
            pids = list(labels.index)
            fold_indices_path = os.path.join(self.data_path, f"filtered_labels/train_labels/{self.mode}_fold_{self.fold}.pkl")
            with open(fold_indices_path, "rb") as f:
                indices = pickle.load(f)
            self.pids = [pids[i] for i in indices]
            self.labels = labels.loc[self.pids]

        self.mean_std = self._load_mean_std()

    def _load_mean_std(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Loads or computes mean and std for normalization.
        Returns:
            Dictionary with mean and std for CT and PET.
        """
        mean_std_path = os.path.join(self.data_path, self.bbox_or_centroid, "train_data", f"mean_std_fold_{self.fold}.pt")
        try:
            return torch.load(mean_std_path)
        except FileNotFoundError:
            print("Calculating normalization factors.")
            if self.mode == "train":
                mean_std = self._get_mean_std()
                torch.save(mean_std, mean_std_path)
                return mean_std
            else:
                raise RuntimeError("Normalization factors not found and not in training mode.")

    def _get_mean_std(self) -> Dict[str, Dict[str, float]]:
        """
        Computes mean and std for CT and PET over the dataset.
        Returns:
            Dictionary with mean and std for CT and PET.
        """
        mean = {"ct": [], "pt": []}
        std = {"ct": [], "pt": []}
        for pid in self.pids:
            ct = torch.load(os.path.join(self.data_path, self.bbox_or_centroid, "train_data", "crops_CT", f"{pid}.pt"), weights_only=False)
            pt = torch.load(os.path.join(self.data_path, self.bbox_or_centroid, "train_data", "crops_PET", f"{pid}.pt"), weights_only=False)
            mean["ct"].append(ct.mean().item())
            mean["pt"].append(pt.mean().item())
            std["ct"].append(ct.std().item())
            std["pt"].append(pt.std().item())
        return {
            "mean": {"ct": torch.tensor(mean["ct"]).mean().item(), "pt": torch.tensor(mean["pt"]).mean().item()},
            "std": {"ct": torch.tensor(std["ct"]).mean().item(), "pt": torch.tensor(std["pt"]).mean().item()}
        }

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.pids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads and returns a sample.
        Args:
            idx: Index of the sample.
        Returns:
            Dictionary with pid, crop, and labels.
        """
        pid = self.pids[idx]
        label = self.labels.loc[pid]
        crops = self._load_crops(pid)
        crops = self._normalize_crops(crops)
        crops = self._apply_monai_3D_transforms(crops)
        crops = self._extract_slice(crops)
        if self.modality == "Merged":
            crop_merged = 0.5 * crops["CT"] + 0.5 * crops["PET"]
            return {"pid": pid, "crops": crop_merged, "labels": label.to_numpy().astype(float)}
        return {"pid": pid, "crops": crops[self.modality], "labels": label.to_numpy().astype(float)}

    def _load_crops(self, pid: str) -> Dict[str, torch.Tensor]:
        """
        Loads CT and/or PET crops for a patient.
        Uses format: (C, H, W, Z)
        Args:
            pid: Patient ID.
        Returns:
            Dictionary with crops.
        """
        crops = {}
        if self.mode == "test":
            img_dir = os.path.join(self.data_path, self.bbox_or_centroid, "test_data")
        else:
            img_dir = os.path.join(self.data_path, self.bbox_or_centroid, "train_data")
        if self.modality in ["CT", "Merged"]:
            ct_crop = torch.load(os.path.join(img_dir, f"crops_CT/{pid}.pt"), weights_only=False)
            crops["CT"] = ct_crop.unsqueeze(dim=0)
        if self.modality in ["PET", "Merged"]:
            pet_crop = torch.load(os.path.join(img_dir, f"crops_PET/{pid}.pt"), weights_only=False)
            crops["PET"] = pet_crop.unsqueeze(dim=0)
        return crops

    def _normalize_crops(self, crops: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalizes crops using mean and std.
        Args:
            crops: Dictionary of crops.
        Returns:
            Normalized crops.
        """
        if "CT" in crops:
            crops["CT"] = TF.normalize(crops["CT"], mean=self.mean_std["mean"]["ct"], std=self.mean_std["std"]["ct"])
        if "PET" in crops:
            crops["PET"] = TF.normalize(crops["PET"], mean=crops["PET"].mean(), std=crops["PET"].std())
            #crops["PET"] = TF.normalize(crops["PET"], mean=self.mean_std["mean"]["pt"], std=self.mean_std["std"]["pt"])
        return crops

    # def _apply_transforms(self, crops: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """
    #     Applies optional transforms to crops.
    #     Args:
    #         crops: Dictionary of crops.
    #     Returns:
    #         Transformed crops.
    #     """
    #     if self.transform is not None:
    #         for mod in crops.keys():
    #             crops[mod] = self.transform(crops[mod])
    #     return crops
    
    def _apply_monai_3D_transforms(self, crops: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Applies optional MONAI 3D transforms to crops.
        Args:
            crops: Dictionary of crops.
        Returns:
            Transformed crops.
        """
        if self.monai_3D_transform is not None:
            crops = self.monai_3D_transform(crops)
        return crops    

    def _extract_slice(self, crops: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extracts a random slice from the center region of the crop.
        Args:
            crops: Dictionary of crops.
        Returns:
            Crops with extracted slice.
        """
        key = list(crops.keys())[0]
        C, Z, H, W = crops[key].shape
        center = Z // 2
        start = max(center - 5, 0)
        end = min(center + 5, Z - 1)
        slices = np.linspace(start, end, min(10, end - start + 1)).astype(int)
        slice_ = random.choice(slices)
        for mod in crops:
            crop = crops[mod][:, slice_, :, :]
            crop = TF.resize(crop, size=(224,224))
            crops[mod] = crop.repeat(3, 1, 1)
        return crops