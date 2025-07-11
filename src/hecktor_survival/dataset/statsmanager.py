from typing import List, Dict
from pathlib import Path
import torch
import os


class StatsManager:
    """
    Handles computation and loading of image normalization stats and RFS scaler.
    """
    def __init__(self, data_path: Path, bbox_or_centroid: str, fold: int):
        self.data_path = data_path
        self.bbox_or_centroid = bbox_or_centroid
        self.fold = fold

    @property
    def image_stats_path(self) -> Path:
        return self.data_path / self.bbox_or_centroid / "train_data" / f"mean_std_fold_{self.fold}.pt"

    def load_or_compute_image_stats(self, pids: List[str], mode: str) -> Dict[str, Dict[str, float]]:
        if self.image_stats_path.is_file():
            return torch.load(self.image_stats_path)
        if mode != 'train':
            raise RuntimeError("Image stats missing and cannot compute outside training mode.")
        stats = self._compute_image_stats(pids)
        tmp = self.image_stats_path.with_suffix('.tmp')
        torch.save(stats, tmp)
        os.replace(tmp, self.image_stats_path)
        return stats

    def _compute_image_stats(self, pids: List[str]) -> Dict[str, Dict[str, float]]:
        means, stds = {'CT': [], 'PET': []}, {'CT': [], 'PET': []}
        for pid in pids:
            for mod in means.keys():
                path = self.data_path / self.bbox_or_centroid / 'train_data' / f'crops_{mod}' / f'{pid}.pt'
                vol = torch.load(path)
                means[mod].append(vol.mean().item())
                stds[mod].append(vol.std().item())
        return {
            'mean': {m: torch.tensor(means[m]).mean().item() for m in means},
            'std':  {m: torch.tensor(stds[m]).mean().item() for m in stds}
        }
    