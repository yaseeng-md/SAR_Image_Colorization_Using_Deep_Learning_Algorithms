import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from Pairing_Images import PairFinder


class PairedImageDataset(Dataset):
    """
    PyTorch Dataset class for paired image loading and preprocessing.

    Attributes:
    -----------
    s1_files : list
        List of input image filenames.
    s2_files : list
        List of output image filenames.
    s1_dir : Path
        Directory path for input images.
    s2_dir : Path
        Directory path for output images.
    transform : torchvision.transforms
        Image transformations.
    """

    def __init__(self, s1_dir, s2_dir, subset_name, save_dataframe=False, image_size=256):
        super().__init__()

        self.s1_dir = Path(s1_dir)
        self.s2_dir = Path(s2_dir)

        # Find the matching image pairs using PairFinder
        pair_finder = PairFinder()
        self.s1_files, self.s2_files = pair_finder.find_the_pairs(self.s1_dir, self.s2_dir, subset_name, save_dataframe)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.s1_files)

    def __getitem__(self, idx):
        s1_path = self.s1_dir / self.s1_files[idx]
        s2_path = self.s2_dir / self.s2_files[idx]

        input_image = Image.open(s1_path).convert("RGB")
        output_image = Image.open(s2_path).convert("RGB")

        input_tensor = self.transform(input_image)
        output_tensor = self.transform(output_image)

        return input_tensor, output_tensor
