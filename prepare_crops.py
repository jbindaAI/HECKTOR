# Script to prepare hxhxh crops of CT and PET scans.

from pathlib import Path
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch

# Defining data paths:
MODE = "test_data"

if MODE == "test_data":
    imagePath = '/home/dzban112/HECKTOR/hecktor2022/hecktor2022_testing/imagesTs_resampled/'
    maskPath = '/home/dzban112/HECKTOR/hecktor2022/hecktor2022_testing/labelsTs_resampled/'
    clinical_df = pd.read_csv("/home/dzban112/HECKTOR/hecktor2022/hecktor2022_testing/hecktor2022_clinical_info_testing.csv")
    endpoint = pd.read_csv("/home/dzban112/HECKTOR/hecktor2022/hecktor2022_testing/hecktor2022_endpoint_testing.csv")
    save_path = '/home/dzban112/HECKTOR/Data/test_data/'
elif MODE == "train_data":
    imagePath = '/home/dzban112/HECKTOR/hecktor2022/hecktor2022_training/imagesTr_resampled/'
    maskPath = '/home/dzban112/HECKTOR/hecktor2022/hecktor2022_training/labelsTr_resampled/'
    clinical_df = pd.read_csv("/home/dzban112/HECKTOR/hecktor2022/hecktor2022_training/hecktor2022_patient_info_training.csv")
    endpoint = pd.read_csv("/home/dzban112/HECKTOR/hecktor2022/hecktor2022_training/hecktor2022_patient_endpoint_training.csv")
    save_path = '/home/dzban112/HECKTOR/Data/train_data/'

# Extracting patients IDs:
ids = clinical_df['PatientID'].to_list()

# Useful functions:
def load_image(id, maskPath, imagePath):
    maskPath = os.path.join(maskPath, id + '.nii.gz')
    ctPath = os.path.join(imagePath, id + '__CT.nii.gz')
    petPath = os.path.join(imagePath, id + '__PT.nii.gz')
    mask = sitk.ReadImage(maskPath)
    CT = sitk.ReadImage(ctPath)
    PET = sitk.ReadImage(petPath)
    mask = sitk.GetArrayFromImage(mask)
    CT = sitk.GetArrayFromImage(CT)
    PET = sitk.GetArrayFromImage(PET)
    return CT, PET, mask


def extract_nodule(patient_id, margin, maskPath, imagePath):
    CT, PET, mask = load_image(patient_id, maskPath, imagePath)
    if 1 not in np.unique(mask):
        raise Exception(f"Mask contains only following labels: {np.unique(mask)}")
    segmentation = np.where(mask == 1)
    bbox = 0, 0, 0, 0
    if len(segmentation) != 0:
        z_min = int(np.min(segmentation[0]))
        z_max = int(np.max(segmentation[0]))
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[2]))
        y_max = int(np.max(segmentation[2]))   
        bbox = [(z_min, z_max), (x_min, x_max), (y_min, y_max)]
        #Center of bbox is computed by averaging extreme indexes on each of dimmensions.
        cbbox = [round((bbox[i][0]+bbox[i][1])/2) for i in range(3)]
        slices = (
                slice(max(cbbox[0]-margin, 0), min(cbbox[0]+margin, CT.shape[0]-1)),
                slice(max(cbbox[1]-margin, 0), min(cbbox[1]+margin, CT.shape[1]-1)),
                slice(max(cbbox[2]-margin, 0) , min(cbbox[2]+margin, CT.shape[2]-1))
             )
        #Cropping a nodule volume from res_data by defined slices tupple.
        crop_CT = torch.Tensor(CT[slices])
        crop_PET = torch.Tensor(PET[slices])
        crop_mask = torch.Tensor(mask[slices])
        if crop_CT[50,:,:].sum().item()==0:
            raise Exception("Defective CT image. Filled with zeros at the central slice.")
        if crop_PET[50,:,:].sum().item()==0:
            raise Exception("Defective PET image. Filled with zeros at the central slice.")
    return crop_CT, crop_PET, crop_mask


Path(f"{save_path}/crops_CT").mkdir(parents=True, exist_ok=True)
Path(f"{save_path}/crops_PET").mkdir(parents=True, exist_ok=True)
Path(f"{save_path}/crops_mask").mkdir(parents=True, exist_ok=True)


## DATA PREPARATION

args = {
    "IDs": ids,
    "margin": 50,
    "save_path": save_path,
    "maskPath": maskPath, 
    "imagePath": imagePath
}

def prepare_data(IDs, margin, save_path, maskPath, imagePath):
    for patient_id in IDs:
        try:
            crop_CT, crop_PET, crop_mask = extract_nodule(patient_id, margin, maskPath, imagePath)
            torch.save(crop_CT.clone(), f"{save_path}/crops_CT/{patient_id}.pt")
            torch.save(crop_PET.clone(), f"{save_path}/crops_PET/{patient_id}.pt")
            torch.save(crop_mask.clone(), f"{save_path}/crops_mask/{patient_id}.pt")
        except Exception as e:
            print(f"Error when processing patient: {patient_id}")
            print(e)
    return "Done"

if __name__ == "__main__":
    prepare_data(**args)
