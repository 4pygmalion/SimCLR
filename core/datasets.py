import random
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torchvision.transforms import ToTensor, Compose


class SimCLRDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: List[str],
        augmentations: List[callable],
        transforms: Compose,
        device: str,
    ):
        self.image_paths = image_paths
        self.augmentations = augmentations
        self.transforms = transforms
        self.device = device

    def __getitem__(self, index) -> Tuple:
        x: Image.Image = Image.open(self.image_paths[index])
        t1, t2 = random.sample(self.augmentations, k=2)
        xi = t1(x)
        xj = t2(x)

        return self.transforms(xi).to(self.device), self.transforms(xj).to(self.device)

    def __len__(self):
        return len(self.image_paths)
