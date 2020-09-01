import os

RGB_ONLY_3_CHANNELS = "\\saved_models\\RGB_ONLY_3_CHANNELS\\rgb_s_cnn_RGB_bin_model.pt"
SPECTRAL_ONLY_7_CHANNELS = "\\saved_models\\SPECTRAL_ONLY_7_CHANNELS\\rgb_s_cnn_with_spectral_7_bin.pt"
SPECTRAL_RGB_10_CHANNELS = "\\saved_models\\SPECTRAL_RGB_10_CHANNELS\\rgb_s_cnn_with_spectral_10_bin.pt"

RGB_ONLY_3_CHANNELS_MULTI = "\\saved_models\\RGB_ONLY_3_CHANNELS_MULTI\\rgb_s_cnn_RGB_multi_model.pt"
SPECTRAL_ONLY_3_CHANNELS_MULTI = "\\saved_models\\SPECTRAL_ONLY_3_CHANNELS_MULTI\\rgb_s_cnn_with_spectral_3_multi.pt"
SPECTRAL_ONLY_7_CHANNELS_MULTI = "\\saved_models\\SPECTRAL_ONLY_7_CHANNELS_MULTI\\rgb_s_cnn_with_spectral_7_multi.pt"
SPECTRAL_RGB_10_CHANNELS_MULTI = "\\saved_models\\SPECTRAL_RGB_10_CHANNELS_MULTI\\rgb_s_cnn_with_spectral_10_multi.pt"

# TODO: In final version make them as parameters to function. ARGS, for example

RGB_CHANNELS = {
    "num": 3,
    "channels": {
        "R": True, "G": True, "B": True, "S1": False, "S2": False, "S3": False, "S4": False, "S5": False,
        "S6": False, "S7": False
    },
    "model_path": os.path.dirname(os.path.abspath(__file__)) + RGB_ONLY_3_CHANNELS,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\binary\simulation_dataset.csv",
    "msg": 'Start training: RGB Channels Binary-Model',
    "results_folder": os.path.dirname(os.path.abspath(__file__)) + "\\saved_models\\RGB_ONLY_3_CHANNELS"
}

SPECTRAL_CHANNELS = {
    "num": 7,
    "channels": {
        "R": False, "G": False, "B": False, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
        "S6": True, "S7": True
    },
    "model_path": os.path.dirname(os.path.abspath(__file__)) + SPECTRAL_ONLY_7_CHANNELS,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\binary\simulation_dataset.csv",
    "msg": 'Start training: Spectral Channels Binary-Model'
}

ALL_CHANNELS = {
    "num": 10,
    "channels": {
        "R": True, "G": True, "B": True, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
        "S6": True, "S7": True
    },
    "model_path": os.path.dirname(os.path.abspath(__file__)) + SPECTRAL_RGB_10_CHANNELS,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\binary\simulation_dataset.csv",
    "msg": 'Start training: All Channels Binary-Model'
}

RGB_CHANNELS_MULTI = {
    "num": 3,
    "channels": {
        "R": True, "G": True, "B": True, "S1": False, "S2": False, "S3": False, "S4": False, "S5": False,
        "S6": False, "S7": False
    },
    "model_path": os.path.dirname(os.path.abspath(__file__)) + RGB_ONLY_3_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_dataset.csv",
    "msg": 'Start training: RGB Channels Multi-Model',
    "results_folder": os.path.dirname(os.path.abspath(__file__)) + "\\saved_models\\RGB_ONLY_3_CHANNELS_MULTI"
}

SPECTRAL_CHANNELS_MULTI = {
    "num": 7,
    "channels": {
        "R": False, "G": False, "B": False, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
        "S6": True, "S7": True
    },
    "model_path": os.path.dirname(os.path.abspath(__file__)) + SPECTRAL_ONLY_7_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_spectral_dataset.csv",
    "msg": 'Start training: Spectral Channels Multi-Model'
}

SPECTRAL_3_CHANNELS_MULTI = {
    "num": 3,
    "channels": {
        "R": False, "G": False, "B": False, "S1": True, "S2": True, "S3": True, "S4": False, "S5": False,
        "S6": False, "S7": False
    },
    "model_path": os.path.dirname(os.path.abspath(__file__)) + SPECTRAL_ONLY_3_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_spectral_dataset.csv",
    "msg": 'Start training: Spectral 3 Channels Multi-Model'
}

ALL_CHANNELS_MULTI = {
    "num": 10,
    "channels": {
        "R": True, "G": True, "B": True, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
        "S6": True, "S7": True
    },
    "model_path": os.path.dirname(os.path.abspath(__file__)) + SPECTRAL_RGB_10_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_dataset.csv",
    "msg": 'Start training: All Channels Multi-Model'
}
