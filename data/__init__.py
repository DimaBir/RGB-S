import os

from torch import nn

import utils
import torch
import random
import kornia.augmentation as K
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


class BananaRustsOneDataset(Dataset):
    """Banana Leaves dataset.(With rust decease)"""

    def __init__(self, csv_file, channels_vector: dict, case="train", root_dir=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            channels_vector (dictionary): represents hot-one vector in order to choose which channel will participate.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.channels_vector = channels_vector
        df = pd.read_csv(csv_file)
        case = case.lower()

        train_set = df.loc[df['case'] == 'train']
        test_set = df.loc[df['case'] == 'test']
        validation_set = df.loc[df['case'] == 'validation']

        self.df = train_set

        if case == "train":
            self.df = train_set
        elif case == "test":
            self.df = test_set
        elif case == "validation":
            self.df = validation_set

        self.root_dir = root_dir
        self.transform = transform

    def load_data(self):
        return self

    def __len__(self):
        return len(self.df)

    def fill_the_channels_to_tensor(self, tensor, channels) -> torch.Tensor:
        j = 0
        for channel in self.channels_vector:
            # Check what channels to add to tensor
            if self.channels_vector[channel]:
                tensor[..., j] = channels[j]
                j = j + 1
        return tensor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Create tensor with chosen channels from channels_vector
        row = self.df.iloc[idx]
        new_size = (128, 128)

        channels = []
        if self.channels_vector["R"]:
            r_ch = np.asarray(Image.open(row['image_R']).resize(new_size), dtype=np.uint8)
            channels.append(r_ch)
        if self.channels_vector["G"]:
            g_ch = np.asarray(Image.open(row['image_G']).resize(new_size), dtype=np.uint8)
            channels.append(g_ch)
        if self.channels_vector["B"]:
            b_ch = np.asarray(Image.open(row['image_B']).resize(new_size), dtype=np.uint8)
            channels.append(b_ch)
        if self.channels_vector["S1"]:
            s1_ch = np.asarray(Image.open(row['image_S1']).resize(new_size), dtype=np.uint8)
            channels.append(s1_ch)
        if self.channels_vector["S2"]:
            s2_ch = np.asarray(Image.open(row['image_S2']).resize(new_size), dtype=np.uint8)
            channels.append(s2_ch)
        if self.channels_vector["S3"]:
            s3_ch = np.asarray(Image.open(row['image_S3']).resize(new_size), dtype=np.uint8)
            channels.append(s3_ch)
        if self.channels_vector["S4"]:
            s4_ch = np.asarray(Image.open(row['image_S4']).resize(new_size), dtype=np.uint8)
            channels.append(s4_ch)
        if self.channels_vector["S5"]:
            s5_ch = np.asarray(Image.open(row['image_S5']).resize(new_size), dtype=np.uint8)
            channels.append(s5_ch)
        if self.channels_vector["S6"]:
            s6_ch = np.asarray(Image.open(row['image_S6']).resize(new_size), dtype=np.uint8)
            channels.append(s6_ch)
        if self.channels_vector["S7"]:
            s7_ch = np.asarray(Image.open(row['image_S7']).resize(new_size), dtype=np.uint8)
            channels.append(s7_ch)

        # Create an empty tensor
        channels_num = sum(self.channels_vector[key] is True for key in self.channels_vector)
        empty_tensor_as_np_array = np.zeros((128, 128, channels_num))

        # Fill it with channels
        tensor_as_np_array = self.fill_the_channels_to_tensor(empty_tensor_as_np_array, channels)

        # Normalize pixels
        image = np.uint8(tensor_as_np_array[:, :, :]) / 255

        # Transform all tensor
        if self.transform:
            image = self.transform(image)

            # Kornia`s transform
            k_transform = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=6.0),
                # Additional transformations
                # K.RandomCrop(size=(128, 128), padding=(10, 10)),
                # K. RandomVerticalFlip(p=0.5)
            )

            # Augmentation
            image = k_transform(image)[0]

        return image, row['label'], os.path.split(row['image_path'])[1]


def create_dataset(path_to_csv: object, case: object, channels_vector: dict, transform: object) -> object:
    """Create a dataset given the option."""

    data_loader = BananaRustsOneDataset(csv_file=path_to_csv, case=case, transform=transform,
                                        channels_vector=channels_vector)
    dataset = data_loader.load_data()
    return dataset


def data_transform(mean_tuple=None, std_tuple=None):
    # TODO: Add number of channels as tuple and mean and std count
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),
        # transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize(mean_tuple, std_tuple)  # Spectral
    ])

    return transform
