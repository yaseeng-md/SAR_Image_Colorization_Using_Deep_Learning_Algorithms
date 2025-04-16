import os
import numpy as np
import pandas as pd
from pathlib import Path

class PairFinder: 
    """
    Finds and pairs corresponding images from two directories (s1 and s2).
    
    Assumes that "s1" in a filename can be replaced with "s2" to locate the corresponding image in the other folder.
    """

    def __init__(self):
        pass

    def find_the_pairs(self, s1_img_path, s2_img_path, subset_name, save_dataframe=False):
        """
        Finds corresponding image pairs between the s1 and s2 directories.

        Parameters:
        -----------
        s1_img_path : str
            Path to the directory containing s1 images.
        s2_img_path : str
            Path to the directory containing s2 images.
        subset_name : str
            Name of the subset (used for saving CSV).
        save_dataframe : bool, optional
            Whether to save the paired data as a CSV file (default is False).

        Returns:
        --------
        tuple of lists
            (s1_files, s2_files) - Lists containing matched filenames.
        """

        self.s1_img_path = Path(s1_img_path)
        self.s2_img_path = Path(s2_img_path)

        s1_img_files = os.listdir(self.s1_img_path)
        s2_img_files = set(os.listdir(self.s2_img_path))  # Use a set for fast lookups

        pair_dataframe = []

        for s1_image in s1_img_files:
            split_path = s1_image.split("_")
            if "s1" in split_path:
                split_path[split_path.index("s1")] = "s2"
                s2_image = "_".join(split_path)

                if s2_image in s2_img_files:
                    pair_dataframe.append([s1_image, s2_image])

        df = pd.DataFrame(pair_dataframe, columns=["s1", "s2"])

        if save_dataframe:
            csv_path = Path(r"C:\Users\gandl\Documents\Python Files\Projects\SAR Image Colorization\Pairs CSV")
            csv_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            df.to_csv(csv_path / f"{subset_name}.csv", index=False)

        return df["s1"].tolist(), df["s2"].tolist()
