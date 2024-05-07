import pandas as pd
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
import numpy as np
import random


class CropData(Dataset):
    """
    Dataset for 100x100x100 crops and RFS/Relapse labels
    """

    def __init__(
        self,
        crops_path="/home/dzban112/HECKTOR/Data/",
        train_labels_file="/home/dzban112/HECKTOR/hecktor2022/hecktor2022_training/hecktor2022_patient_endpoint_training.csv",
        test_labels_file="/home/dzban112/HECKTOR/hecktor2022/hecktor2022_testing/hecktor2022_endpoint_testing.csv",
        transform = None,
        train=True,
    ):
        self.transform = transform
        self.train = train
        self.crops_path = crops_path
    
        # Load RFS scores
        self.train_labels = pd.read_csv(train_labels_file)
        self.test_labels = pd.read_csv(test_labels_file)

        self.train_pids = list(self.train_labels["PatientID"])
        self.test_pids = list(self.test_labels["PatientID"])

        self.train_labels.set_index("PatientID", inplace=True)
        self.test_labels.set_index("PatientID", inplace=True)

        # Change the order of RFS and Relapse columns, as expected by CoxPHLoss
        self.train_labels = self.train_labels[["RFS", "Relapse"]]
        self.test_labels = self.test_labels[["RFS", "Relapse"]]

        def get_mean_std():
            """
            Get mean and std of CT & PET images in training set
            """            
            mean = {"ct": [], "pt": []}
            std = {"ct": [], "pt": []}
            for pid in self.train_pids:
                ct = torch.load(f"{self.crops_path}/train_data/crops_CT/{pid}.pt")
                pt = torch.load(f"{self.crops_path}/train_data/crops_PET/{pid}.pt")
                mean["ct"].append(ct.mean())
                mean["pt"].append(pt.mean())
                std["ct"].append(ct.std())
                std["pt"].append(pt.std())
            return {
                "mean": {"ct": torch.tensor(mean["ct"]).mean(), "pt": torch.tensor(mean["pt"]).mean()},
                "std": {"ct": torch.tensor(std["ct"]).mean(), "pt": torch.tensor(std["pt"]).mean()}
                }
        
        # If the file mean_std_CT.pt exists, load it 
        # Otherwise, calculate mean and std and save it
        try:
            self.mean_std = torch.load("mean_std.pt")
        except FileNotFoundError:
            print("calculating")
            mean_std = get_mean_std()
            self.mean_std = mean_std
            torch.save(mean_std, "mean_std.pt")

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, idx):
        if self.train:
            pid = self.train_pids[idx]
            labels = self.train_labels.iloc[idx]
            crop_CT = torch.load(f"{self.crops_path}/train_data/crops_CT/{pid}.pt")
            crop_PT = torch.load(f"{self.crops_path}/train_data/crops_PET/{pid}.pt")
        else:
            pid = self.test_pids[idx]
            labels = self.test_labels.iloc[idx]
            crop_CT = torch.load(f"{self.crops_path}/test_data/crops_CT/{pid}.pt")
            crop_PT = torch.load(f"{self.crops_path}/test_data/crops_PET/{pid}.pt")

        # Standard seems to be to normalize CT with dataset mean and std
        # and normalize PT image with its own mean and std

        crop_CT =  TF.normalize(
            crop_CT, mean=self.mean_std["mean"]["ct"], std=self.mean_std["std"]["ct"]
        )
        crop_PT = TF.normalize(
            crop_PT, mean=crop_PT.mean(), std=crop_PT.std()
        )

        slices = np.linspace(13, 18, 6).astype(int)
        slice_ = random.choice(slices) # one of the middle slices [14, 15, 16, 17, 18] is chosen.
        crop_CT = crop_CT[slice_, :, :].unsqueeze(0)
        crop_PT = crop_PT[slice_, :, :].unsqueeze(0)
        
        # As ViT model requires 3 color channels,
        # code below makes 2 more channels by coping original channel.
        crop_CT = crop_CT.repeat(3,1,1)
        crop_PT = crop_PT.repeat(3,1,1)
        
        # Apply transforms
        if self.transform:
            crop_CT = self.transform(crop_CT)
            crop_PT = self.transform(crop_PT)

        return {
            "pid": pid,
            "crop_CT": crop_CT,
            "crop_PET": crop_PT,
            "labels": labels.to_numpy().astype(float),
        }
