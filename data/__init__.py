import os

from torch import nn

import utils
import torch
import random
import kornia
import kornia.augmentation as K
import numpy as np
import pandas as pd


from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


class BananaRustsOneDataset(Dataset):
    """Banana Leaves dataset.(With rust decease)"""

    def __init__(self, csv_file, channels_vector: dict, case="Train", root_dir=None, transform=None):
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

        train_set = df.loc[df['case'] == 'Train']
        test_set = df.loc[df['case'] == 'Test']
        validation_set = df.loc[df['case'] == 'Validation']

        self.df = train_set

        if case == "Train":
            self.df = train_set
        elif case == "Test":
            self.df = test_set
        elif case == "Validation":
            self.df = validation_set

        self.root_dir = root_dir
        self.transform = transform

    def load_data(self):
        return self

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        new_size = (128, 128)

        # Create tensor with chosen channels from channels_vector
        # TODO: To make better function function() UGH

        if self.channels_vector["R"]:
            r_channel = np.asarray(Image.open(row['image_R']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["G"]:
            g_channel = np.asarray(Image.open(row['image_G']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["B"]:
            b_channel = np.asarray(Image.open(row['image_B']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["S1"]:
            s1_channel = np.asarray(Image.open(row['image_S1']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["S2"]:
            s2_channel = np.asarray(Image.open(row['image_S2']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["S3"]:
            s3_channel = np.asarray(Image.open(row['image_S3']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["S4"]:
            s4_channel = np.asarray(Image.open(row['image_S4']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["S5"]:
            s5_channel = np.asarray(Image.open(row['image_S5']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["S6"]:
            s6_channel = np.asarray(Image.open(row['image_S6']).resize(new_size), dtype=np.uint8)
        if self.channels_vector["S7"]:
            s7_channel = np.asarray(Image.open(row['image_S7']).resize(new_size), dtype=np.uint8)

        # Tensor preparation
        len = sum(self.channels_vector[key] is True for key in self.channels_vector)
        conc_array = np.zeros((128, 128, len))

        #TODO: Urgent! Please make BETTER solution
        i = 0
        j = 0
        for key in self.channels_vector:
            if self.channels_vector[key] and i == 0:
                conc_array[..., j] = r_channel
                j = j + 1
            elif self.channels_vector[key] and i == 1:
                conc_array[..., j] = g_channel
                j = j + 1
            elif self.channels_vector[key] and i == 2:
                conc_array[..., j] = b_channel
                j = j + 1
            elif self.channels_vector[key] and i == 3:
                conc_array[..., j] = s1_channel
                j = j + 1
            elif self.channels_vector[key] and i == 4:
                conc_array[..., j] = s2_channel
                j = j + 1
            elif self.channels_vector[key] and i == 5:
                conc_array[..., j] = s3_channel
                j = j + 1
            elif self.channels_vector[key] and i == 6:
                conc_array[..., j] = s4_channel
                j = j + 1
            elif self.channels_vector[key] and i == 7:
                conc_array[..., j] = s5_channel
                j = j + 1
            elif self.channels_vector[key] and i == 8:
                conc_array[..., j] = s6_channel
                j = j + 1
            elif self.channels_vector[key] and i == 9:
                conc_array[..., j] = s7_channel
                j = j + 1

            i = i + 1
        # Normalize
        image = np.uint8(conc_array[:, :, :])/255

        # Transform
        if self.transform:
            image = self.transform(image)

            # Kornia`s transform
            k_transform = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=5.0),
                K.RandomCrop(size=(128, 128), padding=(10, 10))
            )

            # Augmentation
            # a = image[0].numpy()
            # plt.imshow(a)
            # plt.show()
            image = k_transform(image)[0]  # why does it adds 1-d after transform? [3, 128, 128] -> [1, 3, 128, 128]
            # a = image[0].numpy()
            # plt.imshow(a)
            # plt.show()

        return image, row['label'], os.path.split(row['image_path'])[1]


def create_dataset(path_to_csv: object, case: object, channels_vector: dict, transform: object) -> object:
    """Create a dataset given the option."""

    data_loader = BananaRustsOneDataset(csv_file=path_to_csv, case=case, transform=transform,
                                        channels_vector=channels_vector)
    dataset = data_loader.load_data()
    return dataset


def data_transform():
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),
        # transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    return transform