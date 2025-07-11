"""
Script 2: Bounding-box cropping with padding
Extracts region tightly around the lesion bounding box plus optional padding.
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from typing import List, Tuple

# ==== Configuration ====
MODE = "train_data"  # or "train_data"
PADDING = 10  # voxels around the lesion bbox

if MODE == "test_data":
    IMAGE_DIR = './data/raw_data/hecktor2022/hecktor2022_testing/imagesTs_resampled/'
    MASK_DIR  = './data/raw_data/hecktor2022/hecktor2022_testing/labelsTs_resampled/'
    CLINICAL_CSV = './data/raw_data/hecktor2022/hecktor2022_testing/hecktor2022_clinical_info_testing.csv'
    SAVE_DIR = './data/bbox/test_data/'
else:
    IMAGE_DIR = './data/raw_data/hecktor2022/hecktor2022_training/imagesTr_resampled/'
    MASK_DIR  = './data/raw_data/hecktor2022/hecktor2022_training/labelsTr_resampled/'
    CLINICAL_CSV = './data/raw_data/hecktor2022/hecktor2022_training/hecktor2022_patient_info_training.csv'
    SAVE_DIR = './data/bbox/train_data/'

Path(SAVE_DIR, 'crops_CT').mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR, 'crops_PET').mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR, 'crops_mask').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLINICAL_CSV)
PATIENT_IDS = df['PatientID'].astype(str).tolist()

def load_image(patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read CT, PET, and segmentation mask as numpy arrays."""
    mask_itk = sitk.ReadImage(os.path.join(MASK_DIR, f"{patient_id}.nii.gz"))
    ct_itk   = sitk.ReadImage(os.path.join(IMAGE_DIR, f"{patient_id}__CT.nii.gz"))
    pet_itk  = sitk.ReadImage(os.path.join(IMAGE_DIR, f"{patient_id}__PT.nii.gz"))
    return (
        sitk.GetArrayFromImage(ct_itk),
        sitk.GetArrayFromImage(pet_itk),
        sitk.GetArrayFromImage(mask_itk)
    )

def extract_nodule_bbox(patient_id: str, padding: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ct, pet, mask = load_image(patient_id)
    if 1 not in np.unique(mask):
        raise RuntimeError(f"No foreground label=1 in mask for {patient_id}")

    # Find lession voxels
    seg = np.where(mask == 1)
    # Lesion bounding box
    z_min, z_max = int(seg[0].min()), int(seg[0].max())
    y_min, y_max = int(seg[1].min()), int(seg[1].max())
    x_min, x_max = int(seg[2].min()), int(seg[2].max())

    # Apply padding and clamp
    Z, Y, X = ct.shape
    z0p, z1p = max(z_min - padding, 0), min(z_max + padding + 1, Z)
    y0p, y1p = max(y_min - padding, 0), min(y_max + padding + 1, Y)
    x0p, x1p = max(x_min - padding, 0), min(x_max + padding + 1, X)

    crop_ct   = torch.from_numpy(ct[z0p:z1p, y0p:y1p, x0p:x1p]).float()
    crop_pet  = torch.from_numpy(pet[z0p:z1p, y0p:y1p, x0p:x1p]).float()
    crop_mask = torch.from_numpy(mask[z0p:z1p, y0p:y1p, x0p:x1p]).long()

    # Sanity checks
    if crop_ct.sum() == 0:
        raise RuntimeError(f"CT crop all zeros for {patient_id}")
    if crop_mask.sum() == 0:
        raise RuntimeError(f"Mask crop lost lesion for {patient_id}")
    
    return crop_ct, crop_pet, crop_mask


def prepare_bbox(IDs: List[str], padding: int, save_dir: str):
    for pid in IDs:
        try:
            ct_crop, pet_crop, mask_crop = extract_nodule_bbox(pid, padding)
            torch.save(ct_crop, os.path.join(save_dir, 'crops_CT', f"{pid}.pt"))
            torch.save(pet_crop, os.path.join(save_dir, 'crops_PET', f"{pid}.pt"))
            torch.save(mask_crop, os.path.join(save_dir, 'crops_mask', f"{pid}.pt"))
        except Exception as e:
            print(f"Error {pid}: {e}")
    print("Bounding-box cropping done.")

if __name__ == '__main__':
    prepare_bbox(PATIENT_IDS, PADDING, SAVE_DIR)
