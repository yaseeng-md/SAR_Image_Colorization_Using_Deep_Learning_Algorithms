import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
from skimage import color
from Pairing_Images import PairFinder


class PairedImageDataset(Dataset):
    """
    PyTorch Dataset for LAB color space image colorization.
    Inputs: L channel from grayscale image.
    Targets: a,b channels from original color image.
    """

    def __init__(self, s1_dir, s2_dir, subset_name, save_dataframe=False, image_size=256):
        super().__init__()

        self.s1_dir = Path(s1_dir)
        self.s2_dir = Path(s2_dir)

        # Get paired file names
        pair_finder = PairFinder()
        self.s1_files, self.s2_files = pair_finder.find_the_pairs(self.s1_dir, self.s2_dir, subset_name, save_dataframe)

        self.image_size = image_size
        self.resize = transforms.Resize((image_size, image_size))

    def _load_and_convert_lab(self, image_path, is_input=True):
        image = Image.open(image_path).convert('RGB')
        image = self.resize(image)
        image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

        lab = color.rgb2lab(image)

        if is_input:
            L = (lab[:, :, 0:1] / 50.0) - 1.0  # [0,100] → [-1,1]
            return torch.from_numpy(L).permute(2, 0, 1).float()
        else:
            ab = lab[:, :, 1:]
            ab = ((ab + 128) / 255.0) * 2 - 1  # [-128,127] → [-1,1]
            return torch.from_numpy(ab).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.s1_files)

    def __getitem__(self, idx):
        s1_path = self.s1_dir / self.s1_files[idx]
        s2_path = self.s2_dir / self.s2_files[idx]

        input_tensor = self._load_and_convert_lab(s1_path, is_input=True)   # L channel
        output_tensor = self._load_and_convert_lab(s2_path, is_input=False) # a,b channels

        return input_tensor, output_tensor
