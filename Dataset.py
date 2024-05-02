import pandas as pd
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF


class CropData(Dataset):
    """
    Dataset for 100x100x100 crops and RFS/Relapse labels
    """

    def __init__(
        self,
        crops_path="/home/ubuntu/Hecktor/TNM_prediction/crops/",
        train_labels_file="/home/ubuntu/Hecktor/hecktor2022/hecktor2022_training/hecktor2022_patient_endpoint_training.csv",
        test_labels_file="/home/ubuntu/Hecktor/hecktor2022/hecktor2022_testing/hecktor2022_endpoint_testing.csv",
        transform = None,
        channels = ["CT", "PT"],
        train=True,
    ):
        self.transform = transform
        self.train = train
        self.crops_path = crops_path
        self.channels = channels
    
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
            Get mean and std of CT images in training set
            """            
            mean = {"ct": [], "pt": []}
            std = {"ct": [], "pt": []}
            for pid in self.train_pids:
                ct = torch.load(f"{self.crops_path}/training/{pid}__CT.pt")
                pt = torch.load(f"{self.crops_path}/training/{pid}__PT.pt")
                mean["ct"].append(ct.mean())
                mean["pt"].append(pt.mean())
                std["ct"].append(ct.std())
                std["pt"].append(pt.std())
            # return torch.tensor(ct_mean).mean(), torch.tensor(ct_std).mean()
            return {
                "mean": {"ct": torch.tensor(mean["ct"]).mean(), "pt": torch.tensor(mean["pt"]).mean()},
                "std": {"ct": torch.tensor(std["ct"]).mean(), "pt": torch.tensor(std["pt"]).mean()}
                }
        
        # If the file mean_std_CT.pt exists, load it 
        # Otherwise, calculate mean and std and save it
        try:
            self.mean, self.std = torch.load("mean_std_CT.pt")
        except FileNotFoundError:
            print("calculating")
            mean_std = get_mean_std()
            self.mean_std = mean_std
            torch.save((self.ct_mean, self.ct_std), "mean_std_CT.pt")

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, idx):
        if self.train:
            pid = self.train_pids[idx]
            labels = self.train_labels.iloc[idx]
            crop_CT = torch.load(f"{self.crops_path}/training/{pid}__CT.pt")
            crop_PT = torch.load(f"{self.crops_path}/training/{pid}__PT.pt")
        else:
            pid = self.test_pids[idx]
            labels = self.test_labels.iloc[idx]
            crop_CT = torch.load(f"{self.crops_path}/testing/{pid}__CT.pt")
            crop_PT = torch.load(f"{self.crops_path}/testing/{pid}__PT.pt")

        # Standard seems to be to normalize CT with dataset mean and std
        # and normalize PT image with its own mean and std

        crop_CT =  TF.normalize(
            crop_CT, mean=self.mean_std["mean"]["ct"], std=self.mean_std["std"]["ct"]
        )
        crop_PT = TF.normalize(
            crop_PT, mean=crop_PT.mean(), std=crop_PT.std()
        )

        if len(self.channels) == 2:
            crop = torch.stack([crop_CT, crop_PT])
        else:
            if self.channels[0] == "CT":
                crop = crop_CT.unsqueeze(0)
            else:
                crop = crop_PT.unsqueeze(0)
        # Apply transforms
        if self.transform:
            crop = self.transform(crop)

        return {
            "pid": pid,
            "crop_CT": crop_CT,
            "crop_PET": crop_PT,
            "labels": labels.to_numpy().astype(float),
        }
