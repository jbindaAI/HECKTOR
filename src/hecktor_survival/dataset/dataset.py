from typing import Dict, Any, Tuple, Optional, Literal
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import pickle
import torch

from hecktor_survival.dataset.statsmanager import StatsManager


class HECKTOR_Dataset(Dataset):
    """PyTorch Dataset for HECKTOR with refactored stats, paths, transforms, and reproducible slicing."""

    def __init__(
        self,
        data_path: str,
        bbox_or_centroid: Literal['bbox','centroid'],
        fold: int = 1,
        mode: Literal['train','val','test'] = 'train',
        modality: Literal['CT','PET','Merged'] = 'CT',
        normalize_imgs: bool = True,
        transforms: Optional[Dict[str, Any]] = None,
        slice_strategy: Literal['random','center'] = 'random'
    ):
        assert mode in {'train','val','test'}
        assert modality in {'CT','PET','Merged'}
        self.data_path = Path(data_path)
        self.bbox_or_centroid = bbox_or_centroid
        self.fold = fold
        self.mode = mode
        self.modality = modality
        self.normalize_imgs = normalize_imgs
        self.slice_strategy = slice_strategy
        self.transforms = transforms

        # initialize StatsManager
        self.stats_mgr = StatsManager(self.data_path, bbox_or_centroid, fold)

        # load labels
        labels_csv = self.data_path / 'filtered_labels' / ('test_labels' if mode=='test' else 'train_labels') / ('test_labels_filtered.csv' if mode=='test' else 'train_labels_filtered.csv') #f'{mode}_labels_filtered.csv'
        self.labels = pd.read_csv(labels_csv, index_col='PatientID')[['RFS','Relapse']]
        pids_all = self.labels.index.tolist()

        # train/val split
        if mode != 'test':
            fold_file = self.data_path / 'filtered_labels' / 'train_labels' / f'{mode}_fold_{fold}.pkl'
            indices = pickle.load(open(fold_file, 'rb'))
            self.pids = [pids_all[i] for i in indices]
            self.labels = self.labels.loc[self.pids]
        else:
            self.pids = pids_all.copy()

        # normalization stats
        if self.normalize_imgs:
            self.image_stats = self.stats_mgr.load_or_compute_image_stats(self.pids, self.mode)

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.pids[idx]

        # labels
        row = self.labels.loc[pid]
        labels = torch.tensor([row['RFS'], row['Relapse']], dtype=torch.float32)

        # load 3D crops and mask
        crops_3d, mask_3d = self._load_crops(pid)

        # normalize
        if self.normalize_imgs:
            crops_3d = self._normalize_crops(crops_3d)

        # slice extraction
        crops_2d, mask_2d = self._extract_slice(crops_3d, mask_3d)

        # apply transforms
        if self.transforms:
            crops_2d, mask_2d = self.transforms(crops_2d, mask_2d)

        # compute cancer area
        cancer_area = (mask_2d == 1).sum().item() / mask_2d.numel()

        # modality merge if needed
        if self.modality == 'Merged':
            merged = 0.5 * crops_2d['CT'] + 0.5 * crops_2d['PET']
            crops = merged
        else:
            crops = crops_2d[self.modality]

        return {'pid': pid, 'crops': crops, 'mask': mask_2d, 'labels': labels, 'cancer_area': cancer_area}

    def _base_dir(self) -> str:
        return 'test_data' if self.mode=='test' else 'train_data'

    def _load_crops(self, pid: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        base = self._base_dir()
        crops, mask = {}, None
        for mod in ['CT','PET']:
            if self.modality in [mod,'Merged']:
                path = self.data_path / self.bbox_or_centroid / base / f'crops_{mod}' / f'{pid}.pt'
                crops[mod] = torch.load(path).unsqueeze(0)
        mask = torch.load(self.data_path / self.bbox_or_centroid / base / 'crops_mask' / f'{pid}.pt').unsqueeze(0)
        return crops, mask

    def _normalize_crops(self, crops: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for mod, vol in crops.items():
            m, s = self.image_stats['mean'][mod], self.image_stats['std'][mod]
            out[mod] = TF.normalize(vol, mean=[m], std=[s])
        return out

    def _extract_slice(
        self,
        crops: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        _, Z, H, W = next(iter(crops.values())).shape
        center = Z // 2
        slices = list(range(max(center-5,0), min(center+5,Z-1)+1))
        if self.slice_strategy == 'random' and self.mode=='train':
            idx = torch.multinomial(torch.ones(len(slices)), 1).item()
            slice_idx = slices[idx]
        else:
            slice_idx = slices[len(slices)//2]

        crops2d = {mod: TF.resize(vol[:, slice_idx], (224,224)).repeat(3,1,1)
                   for mod, vol in crops.items()}
        mask2d = TF.resize(mask[:, slice_idx], (224,224), interpolation=TF.InterpolationMode.NEAREST)
        return crops2d, mask2d

    def __repr__(self) -> str:
        return (f"<HECKTOR_Dataset mode={self.mode} fold={self.fold} "
                f"modality={self.modality} len={len(self)}" +
                (" normalizing" if self.normalize_imgs else ""))
