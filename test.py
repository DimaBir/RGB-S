import argparse
import os

import train
import torch
import config
import numpy as np
import matplotlib.pyplot as plt

from models import create_model
from torch.utils.data import DataLoader
from data import create_dataset, data_transform
from utils import plot_confusion_matrix, SaveOutput, matplot_plot_images_dict, display_multiple_img, printgradnorm, \
    UnNormalize


def test(show_features_map=False, test_settings=None, test_batch_size=1):
    DICTIONARY = config.SETTINGS[test_settings]
    print(DICTIONARY.msg)

    PATH = DICTIONARY.model_path
    CHANNELS = DICTIONARY.channels
    NUM_OF_CHANNELS = DICTIONARY.num
    DATASET_PATH = DICTIONARY.dataset_path
    RESULTS_FOLDER = DICTIONARY.result_folder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = create_model(num_of_channels=NUM_OF_CHANNELS).to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    test_dataset = create_dataset(path_to_csv=DATASET_PATH,
                                  case="Test",
                                  transform=data_transform(),
                                  channels_vector=CHANNELS)  # create a dataset

    test_dataset_size = len(test_dataset)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    if show_features_map:
        # Register forward hooks
        save_output = SaveOutput()
        hook_handles = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)

    # Test
    y = []
    y_hat = []
    predicted_arr = []

    total = 0
    correct = 0

    print('The number of test images: {}'.format(test_dataset_size))

    for data in test_data_loader:
        images, labels, image_index = data
        [y.append(label.item()) for label in labels]

        images = images.type(torch.cuda.FloatTensor).to(device)
        labels = labels.type(torch.cuda.FloatTensor).to(device)

        outputs = model.predict(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        # Show features map. Output [conv_layer_num], [feature_map_num], [channel_num]
        # TODO: WARNING: Adjust channels for different tests
        if show_features_map:
            plt.imshow(np.moveaxis(images.cpu()[0].numpy()[0:3, :, :], 0, 2))
            plt.show()

            features_list = list(torch.Tensor.cpu(save_output.outputs[0][0]).detach().numpy())
            display_multiple_img(features_list, 8, 8, scale=20)

            features_list = list(torch.Tensor.cpu(save_output.outputs[1][0]).detach().numpy())
            display_multiple_img(features_list, 16, 8, scale=20)

            features_list = list(torch.Tensor.cpu(save_output.outputs[2][0]).detach().numpy())
            display_multiple_img(features_list, 16, 16, scale=20)

            save_output.clear()

        if predicted != labels:  # and predicted.item() == 0:   and img_index <= 40:

            # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            # image = unorm(images.cpu()[0])
            image = images.cpu()[0]

            # We will print each channel separatly to be sure we can plot it to the file
            # PIL can visualize only up to 4 channels
            dir_name_pos = os.path.join(RESULTS_FOLDER, 'false_positive')
            abs_dir_name_pos = os.path.join(dir_name_pos, 'fp_' + str(image_index[0]))

            dir_name_neg = os.path.join(RESULTS_FOLDER, 'false_negative')
            abs_dir_name_neg = os.path.join(dir_name_neg, 'fn_' + str(image_index[0]))

            # TODO: for time saving lets print only first channel but we keep option to
            #       print all images in separate folders in order together information
            # for i in range(NUM_OF_CHANNELS):
            i = 0
            channel_image = image[i, :, :]

            # Save False Positive and False Negative images
            # plt.imshow(np.moveaxis(channel_image, 0, 2))
            plt.imshow(channel_image)

            if predicted.item() == 1:
                if not os.path.exists(abs_dir_name_pos):
                    os.makedirs(abs_dir_name_pos)
                plt.savefig(os.path.join(abs_dir_name_pos, 'fp_' + str(image_index[0]) + f'_{i}.png'))
            else:
                if not os.path.exists(abs_dir_name_neg):
                    os.makedirs(abs_dir_name_neg)
                plt.savefig(os.path.join(abs_dir_name_neg, 'fn_' + str(image_index[0]) + f'_{i}.png'))

        correct += (predicted == labels).sum().item()
        predicted_arr.append(predicted.item())
        [y_hat.append(output[1].item()) for output in outputs]

    # Confusion Matrix
    plot_confusion_matrix(RESULTS_FOLDER, y, predicted_arr)

    y_hat_ones = []
    y_hat_zeros = []

    [y_hat_ones.append(y_hat[i]) for i, j in enumerate(y) if j == 1.]
    [y_hat_zeros.append(y_hat[i]) for i, j in enumerate(y) if j == 0.]

    plt.hist(y_hat_zeros, color='b', bins=100, label='CLass 0', alpha=0.5)
    plt.hist(y_hat_ones, color='r', bins=100, label='Class 1', alpha=0.5)
    plt.gca().set(title='Freq vs Prob of Class [1]; Test Accuracy: %d %%' %
                        (100 * correct / total), ylabel='Frequency')
    plt.legend()
    plt.savefig(RESULTS_FOLDER + '\\test_histo_' + str(int((100 * correct / total))) + '.png')
    plt.show()

    y_hat.clear()
    y.clear()


if __name__ == '__main__':
    # Test settings
    parser = argparse.ArgumentParser(description='Phenomics RGB-T Classifier')
    parser.add_argument('--FM', type=bool, default=False,
                        help='Show first layer feature maps (default: False)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--settings', type=str, choices=config.CHOICES, help='Choose type of model to train')
    args = parser.parse_args()

    test(show_features_map=False, test_settings=args.settings, test_batch_size=args.test_batch_size)
