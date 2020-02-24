import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class BananaRustsOneDataset(Dataset):
    """Banana Leaves dataset.(With rust decease)"""

    def __init__(self, csv_file, case="Train", root_dir=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csv_file)

        train_set = df.loc[df['case'] == 'Train']
        test_set = df.loc[df['case'] == 'Test']
        validation_set = df.loc[df['case'] == 'Validation']

        self.df = train_set

        if case == "Test":
            self.df = test_set
        elif case == "Validation":
            self.df = validation_set

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        r_channel = np.asarray(Image.open(row['image_R']), dtype=np.uint8)
        g_channel = np.asarray(Image.open(row['image_G']), dtype=np.uint8)
        b_channel = np.asarray(Image.open(row['image_B']), dtype=np.uint8)

        s1_channel = np.asarray(Image.open(row['image_S1']), dtype=np.uint8)
        s2_channel = np.asarray(Image.open(row['image_S2']), dtype=np.uint8)
        s3_channel = np.asarray(Image.open(row['image_S3']), dtype=np.uint8)
        s4_channel = np.asarray(Image.open(row['image_S4']), dtype=np.uint8)
        s5_channel = np.asarray(Image.open(row['image_S5']), dtype=np.uint8)
        s6_channel = np.asarray(Image.open(row['image_S6']), dtype=np.uint8)
        s7_channel = np.asarray(Image.open(row['image_S7']), dtype=np.uint8)

        # Tensor preparation
        con_array = np.concatenate((r_channel, g_channel, b_channel,
                                    s1_channel, s2_channel, s3_channel,
                                    s4_channel, s5_channel, s6_channel,
                                    s7_channel))

        con_array = np.reshape(con_array, (128, 128))
        image = Image.fromarray(con_array, 'L')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': row['label']}
