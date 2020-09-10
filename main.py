from test import test
from train import train

TRAIN_SETTINGS = [
    "ALL_CHANNELS",
    "RGB_CHANNELS",
    "ALL_CHANNELS_MULTI",
    "RGB_CHANNELS_MULTI",
    "SPECTRAL_3_CHANNELS_MULTI",
    "SPECTRAL_CHANNELS_MULTI"
]


if __name__ == '__main__':
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--mean_std', type=bool, default=False,
    #                     help='print dataset mean std (default: False)')
    # parser.add_argument('--scheduler', type=bool, default=False,
    #                     help='Use learning rae decay technique (default: False)')
    # parser.add_argument('--L1', type=bool, default=False,
    #                     help='Use L1 regularization (default: False)')
    # parser.add_argument('--batch-size', type=int, default=96,
    #                     help='input batch size for training (default: 96)')
    # parser.add_argument('--test-batch-size', type=int, default=1,
    #                     help='input batch size for testing (default: 1)')
    # parser.add_argument('--epochs', type=int, default=600,
    #                     help='number of epochs to train (default: 600)')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='learning rate (default: 0.001)')
    # parser.add_argument('--fc', type=float, default=0.0,
    #                     help='First Fully Connected layer dropout (default: 0.0)')
    # parser.add_argument('--settings', type=str, default="RGB_CHANNELS", choices=TRAIN_SETTINGS,
    #                     help='Choose type of model to train (default: RGB 3 Channels only)')
    # args = parser.parse_args()
    #
    # train(n_epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size, FC_dp=0.0, train_settings="RGB_CHANNELS", show_feature_maps=False,
    #       print_dataset_mean_std=False, use_scheduler=False, use_L1_regularization=False, show_accuracy=True)
    train(train_settings="SPECTRAL_CHANNELS_MULTI")
    test(train_settings="SPECTRAL_CHANNELS_MULTI")