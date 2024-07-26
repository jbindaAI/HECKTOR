import pandas as pd
from torch.utils.data import Dataset
import torch
import torchvision.transforms.v2.functional as TF
import numpy as np
import random
import os
from typing import Optional, Literal, Union, List
import pickle
from enum import Enum

CWD = os.getcwd()

class Mode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class Modality(Enum):
    CT = "CT"
    PET = "PET"
    BOTH = "both"

class HECKTOR_Dataset(Dataset):
    """
    Dataset for HECKTOR slices with RFS/Relapse labels.
    """

    def __init__(
        self,
        data_path: str = CWD + "/Data/",
        transform: Optional = None,
        fold: Union[int, Literal["all"]] = 1,
        mode: Mode = Mode.TRAIN,
        modality: Modality = Modality.CT,
    ):
        self.mode = mode
        self.fold = fold
        self.data_path = data_path
        self.transform = transform
        self.modality = modality

        self.mode_data = {
            Mode.TRAIN: {
                "pids": "train_pids",
                "labels": "train_labels",
                "data_dir": "train_data"
            },
            Mode.VAL: {
                "pids": "val_pids",
                "labels": "val_labels",
                "data_dir": "train_data"
            },
            Mode.TEST: {
                "pids": "test_pids",
                "labels": "test_labels",
                "data_dir": "test_data"
            }
        }
        self._load_data()

    def _load_data(self):
        mode_info = self.mode_data[self.mode]
        if self.mode == Mode.TRAIN or self.mode == Mode.VAL:
            labels_path = self.data_path + f"{mode_info['data_dir']}/hecktor2022_endpoint_train.csv"
        else:
            labels_path = self.data_path + f"{mode_info['data_dir']}/hecktor2022_endpoint_test.csv"
        labels = pd.read_csv(labels_path)
        pids = labels["PatientID"]
        labels.set_index("PatientID", inplace=True)
        labels = labels[["RFS", "Relapse"]]
        
        if self.mode == Mode.TRAIN or self.mode == Mode.VAL:
            fold_indices_path = self.data_path + f"{mode_info['data_dir']}/{self.mode.value}_fold_{self.fold}.pkl"
            with open(fold_indices_path, "rb") as file:
                indices = pickle.load(file)
            setattr(self, mode_info["pids"], list(pids.iloc[indices]))
            setattr(self, mode_info["labels"], labels.iloc[indices])
        else:
            setattr(self, mode_info["pids"], list(pids))
            setattr(self, mode_info["labels"], labels)

        self._load_mean_std()

    def _load_mean_std(self):
        try:
            self.mean_std = torch.load(self.data_path + f"train_data/mean_std_fold_{self.fold}.pt")
        except FileNotFoundError:
            print("Calculating normalization factors.")
            if self.mode == Mode.TRAIN:
                self.mean_std = self._get_mean_std()
                torch.save(self.mean_std, self.data_path + f"train_data/mean_std_fold_{self.fold}.pt")
            else:
                print("There are not any normalization factors fitted on training data!")

    def _get_mean_std(self) -> dict:
        mean = {"ct": [], "pt": []}
        std = {"ct": [], "pt": []}
        for pid in self.train_pids:
            ct = torch.load(f"{self.data_path}train_data/crops_CT/{pid}.pt")
            pt = torch.load(f"{self.data_path}train_data/crops_PET/{pid}.pt")
            mean["ct"].append(ct.mean())
            mean["pt"].append(pt.mean())
            std["ct"].append(ct.std())
            std["pt"].append(pt.std())
        return {
            "mean": {"ct": torch.tensor(mean["ct"]).mean(), "pt": torch.tensor(mean["pt"]).mean()},
            "std": {"ct": torch.tensor(std["ct"]).mean(), "pt": torch.tensor(std["pt"]).mean()}
        }

    def __len__(self) -> int:
        mode_info = self.mode_data[self.mode]
        return len(getattr(self, mode_info["pids"]))

    def __getitem__(self, idx: int) -> dict:
        pid, labels = self._get_pid_and_labels(idx)
        crops = self._load_crops(pid)
        crops = self._normalize_crops(crops)
        crops = self._apply_transforms(crops)
        crops = self._extract_slice(crops)

        if self.modality == Modality.BOTH:
            crop_stacked = torch.cat((crops["CT"], crops["PET"]), dim=0)
            return {"pid": pid, "crop": crop_stacked, "labels": labels.to_numpy().astype(float)}
        
        return {"pid": pid, "crop": crops[self.modality.value], "labels": labels.to_numpy().astype(float)}

    def _get_pid_and_labels(self, idx: int):
        mode_info = self.mode_data[self.mode]
        return getattr(self, mode_info["pids"])[idx], getattr(self, mode_info["labels"]).iloc[idx]

    def _load_crops(self, pid: str) -> dict:
        crops = {}
        data_directory = "test_data" if self.mode == Mode.TEST else "train_data"
        if self.modality in [Modality.CT, Modality.BOTH]:
            crops["CT"] = torch.load(f"{self.data_path}{data_directory}/crops_CT/{pid}.pt")
        if self.modality in [Modality.PET, Modality.BOTH]:
            crops["PET"] = torch.load(f"{self.data_path}{data_directory}/crops_PET/{pid}.pt")
        return crops

    def _normalize_crops(self, crops: dict) -> dict:
        if "CT" in crops:
            crops["CT"] = TF.normalize(crops["CT"], mean=self.mean_std["mean"]["ct"], std=self.mean_std["std"]["ct"])
        if "PET" in crops:
            #if crops["PET"].sum().item() > 0:
                #crops["PET"] = TF.normalize(crops["PET"], mean=crops["PET"].mean(), std=crops["PET"].std())
            crops["PET"] = TF.normalize(crops["PET"], mean=self.mean_std["mean"]["pt"], std=self.mean_std["std"]["pt"])
        return crops

    def _apply_transforms(self, crops: dict) -> dict:
        for mod in crops.keys():
            crops[mod] = self.transform(crops[mod])
            crops[mod] = TF.resize(crops[mod], size=[224, 224])        
        return crops

    def _extract_slice(self, crops: dict) -> dict:
        slices = np.linspace(45, 55, 10).astype(int)
        slice_ = random.choice(slices)
        for mod in crops:
            crops[mod] = crops[mod][slice_, :, :].unsqueeze(0)
        return crops
