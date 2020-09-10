import os
import random
from utils import ChannelSettings


RGB_ONLY_3_CHANNELS = "\\saved_models\\RGB_ONLY_3_CHANNELS\\rgb_s_cnn_RGB_bin_model.pt"
SPECTRAL_ONLY_7_CHANNELS = "\\saved_models\\SPECTRAL_ONLY_7_CHANNELS\\rgb_s_cnn_with_spectral_7_bin.pt"
SPECTRAL_RGB_10_CHANNELS = "\\saved_models\\SPECTRAL_RGB_10_CHANNELS\\rgb_s_cnn_with_spectral_10_bin.pt"

RGB_ONLY_3_CHANNELS_MULTI = "\\saved_models\\RGB_ONLY_3_CHANNELS_MULTI\\rgb_s_cnn_RGB_multi_model.pt"
SPECTRAL_ONLY_3_CHANNELS_MULTI = "\\saved_models\\SPECTRAL_ONLY_3_CHANNELS_MULTI\\rgb_s_cnn_with_spectral_3_multi.pt"
SPECTRAL_ONLY_7_CHANNELS_MULTI = "\\saved_models\\SPECTRAL_ONLY_7_CHANNELS_MULTI\\rgb_s_cnn_with_spectral_7_multi.pt"
BIG_SPECTRAL_ONLY_7_CHANNELS_MULTI = "\\saved_models\\BIG_SPECTRAL_ONLY_7_CHANNELS_MULTI\\rgb_s_cnn_with_spectral_7_multi.pt"
SPECTRAL_RGB_10_CHANNELS_MULTI = "\\saved_models\\SPECTRAL_RGB_10_CHANNELS_MULTI\\rgb_s_cnn_with_spectral_10_multi.pt"

RGB_CHANNELS = ChannelSettings(name="RGB_CHANNELS_BIN", number_of_channels=3,
                               channels_vector=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               model_path=os.path.dirname(os.path.abspath(__file__)) + RGB_ONLY_3_CHANNELS,
                               dataset_path=r"D:\Github\RGB-S\data\simulation_3\binary\simulation_dataset.csv")

SPECTRAL_CHANNELS = ChannelSettings(name="SPECTRAL_CHANNELS_BIN", number_of_channels=7,
                                    channels_vector=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                    model_path=os.path.dirname(os.path.abspath(__file__)) + SPECTRAL_ONLY_7_CHANNELS,
                                    dataset_path=os.path.join(os.path.dirname(os.path.abspath(
                                        __file__)) + "\\data\\simulation_3\\binary\\simulation_dataset.csv"))

ALL_CHANNELS = ChannelSettings(name="ALL_CHANNELS_BIN", number_of_channels=10,
                               channels_vector=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               model_path=os.path.dirname(os.path.abspath(__file__)) + SPECTRAL_RGB_10_CHANNELS,
                               dataset_path=os.path.join(os.path.dirname(os.path.abspath(
                                   __file__)) + "\\data\\simulation_3\\binary\\simulation_dataset.csv"))

RGB_CHANNELS_MULTI = ChannelSettings(name="RGB_CHANNELS_MULTI", number_of_channels=3,
                                     channels_vector=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                     model_path=os.path.dirname(os.path.abspath(__file__)) + RGB_ONLY_3_CHANNELS_MULTI,
                                     dataset_path=os.path.join(os.path.dirname(os.path.abspath(
                                         __file__)) + "\\data\\simulation_3\\multi\\simulation_dataset.csv"))

SPECTRAL_CHANNELS_MULTI = ChannelSettings(name="SPECTRAL_CHANNELS_MULTI", number_of_channels=7,
                                          channels_vector=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                          model_path=os.path.dirname(
                                              os.path.abspath(__file__)) + SPECTRAL_ONLY_7_CHANNELS_MULTI,
                                          dataset_path=os.path.join(os.path.dirname(os.path.abspath(
                                              __file__)) + "\\data\\simulation_3\\multi\\simulation_dataset.csv"))

BIG_SPECTRAL_CHANNELS_MULTI = ChannelSettings(name="BIG_SPECTRAL_CHANNELS_MULTI", number_of_channels=7,
                                              channels_vector=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                              model_path=os.path.dirname(
                                                  os.path.abspath(__file__)) + BIG_SPECTRAL_ONLY_7_CHANNELS_MULTI,
                                              dataset_path=os.path.join(os.path.dirname(os.path.abspath(
                                                  __file__)) + "\\data\\simulation_5_affine\\binary\\simulation_dataset.csv"))

ALL_CHANNELS_MULTI = ChannelSettings(name="ALL_CHANNELS_MULTI", number_of_channels=10,
                                     channels_vector=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                     model_path=os.path.dirname(
                                         os.path.abspath(__file__)) + SPECTRAL_RGB_10_CHANNELS_MULTI,
                                     dataset_path=os.path.join(os.path.dirname(os.path.abspath(
                                         __file__)) + "\\data\\simulation_3\\multi\\simulation_dataset.csv"))

random_spectral_channels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for ind in random.sample(range(3, 9), 3):
    random_spectral_channels[ind] = 1

SPECTRAL_3_CHANNELS_MULTI = ChannelSettings(name="SPECTRAL_3_CHANNELS_MULTI", number_of_channels=3,
                                            channels_vector=random_spectral_channels,
                                            model_path=os.path.dirname(
                                                os.path.abspath(__file__)) + SPECTRAL_ONLY_3_CHANNELS_MULTI,
                                            dataset_path=os.path.join(os.path.dirname(os.path.abspath(
                                                __file__)) + "\\data\\simulation_3\\multi\\simulation_spectral_dataset.csv"))

######################## ADD HERE ############################

SETTINGS = {
    "RGB_CHANNELS": RGB_CHANNELS,
    "ALL_CHANNELS": ALL_CHANNELS,
    "ALL_CHANNELS_MULTI": ALL_CHANNELS_MULTI,
    "RGB_CHANNELS_MULTI": RGB_CHANNELS_MULTI,
    "SPECTRAL_3_CHANNELS_MULTI": SPECTRAL_3_CHANNELS_MULTI,
    "SPECTRAL_CHANNELS_MULTI": SPECTRAL_CHANNELS_MULTI,
    "BIG_SPECTRAL_CHANNELS_MULTI": BIG_SPECTRAL_CHANNELS_MULTI
}


CHOICES = [
    "RGB_CHANNELS",
    "ALL_CHANNELS",
    "ALL_CHANNELS_MULTI",
    "RGB_CHANNELS_MULTI",
    "SPECTRAL_3_CHANNELS_MULTI",
    "SPECTRAL_CHANNELS_MULTI",
    "BIG_SPECTRAL_CHANNELS_MULTI"
]
