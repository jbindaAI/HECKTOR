import torchvision.transforms.v2.functional as TF
from torchvision.transforms import v2
import torch
import random
from typing import Dict, Tuple


class PairedRandom2DTransform:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size

    def __call__(self, crops: Dict[str, torch.Tensor], mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Random horizontal flip
        if random.random() > 0.5:
            crops = {k: TF.hflip(v) for k, v in crops.items()}
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            crops = {k: TF.vflip(v) for k, v in crops.items()}
            mask = TF.vflip(mask)

        # Random rotation (by 0, 90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            crops = {k: TF.rotate(v, angle) for k, v in crops.items()}
            mask = TF.rotate(mask, angle)

        # Resize
        crops = {k: TF.resize(v, size=(256,256)) for k, v in crops.items()}
        mask = TF.resize(mask, size=(256,256), interpolation=TF.InterpolationMode.NEAREST)

        # Random crop
        i, j, h, w = v2.RandomCrop.get_params(mask, output_size=self.image_size)
        crops = {k: TF.crop(v, i, j, h, w) for k, v in crops.items()}
        mask = TF.crop(mask, i, j, h, w)

        return crops, mask