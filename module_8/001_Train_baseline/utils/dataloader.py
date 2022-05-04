import os
import random
from typing import Tuple
import glob
import numpy as np
import glob


import torch
from torch import Tensor
from torch.utils.data import Dataset

import torchvision

from PIL import Image

SEED = 1236

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class DS(Dataset):
    def __init__(self, images: list, classes: list, resolush: int, use_albu: bool = False):
        self.images = images
        self.classes = classes
        self.use_albu = use_albu
        self.resolush = resolush

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = Image.open(self.images[idx]).convert('RGB')
        img = torchvision.transforms.Resize((self.resolush, self.resolush))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)

        c = os.path.dirname(self.images[idx]).split('/')[-2]

        return img, self.classes.index(c)

    def __len__(self) -> int:
        return len(self.images)