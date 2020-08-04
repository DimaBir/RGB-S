import train
import torch
import config
import numpy as np
import matplotlib.pyplot as plt

from models import create_model
from torch.utils.data import DataLoader
from data import create_dataset, data_transform
from utils import plot_confusion_matrix, SaveOutput, matplot_plot_images_dict, display_multiple_img, printgradnorm
from mpl_toolkits.axes_grid1 import ImageGrid

PATH = train.PATH
CHANNELS = train.CHANNELS
NUM_OF_CHANNELS = train.NUM_OF_CHANNELS
DATASET_PATH = train.DATASET_PATH

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(num_of_channels=NUM_OF_CHANNELS).to(device)

    preatrained_model = torch.load(config.RGB_ONLY_3_CHANNELS)

    # print('\n============== conv1.weight ===================')
    # print("Min: " + str(torch.min(preatrained_model['conv1.weight']).item()))
    # print("Max: " + str(torch.max(preatrained_model['conv1.weight']).item()))
    # print("Mean: " + str(torch.mean(preatrained_model['conv1.weight']).item()))
    #
    # print('\n============== conv2.weight ===================')
    # print("Min: " + str(torch.min(preatrained_model['conv2.weight']).item()))
    # print("Max: " + str(torch.max(preatrained_model['conv2.weight']).item()))
    # print("Mean: " + str(torch.mean(preatrained_model['conv2.weight']).item()))
    #
    # print('\n============== conv3.weight ===================')
    # print("Min: " + str(torch.min(preatrained_model['conv3.weight']).item()))
    # print("Max: " + str(torch.max(preatrained_model['conv3.weight']).item()))
    # print("Mean: " + str(torch.mean(preatrained_model['conv3.weight']).item()))
    #
    # print('\n============== fc1.weight ===================')
    # print("Min: " + str(torch.min(preatrained_model['fc1.weight']).item()))
    # print("Max: " + str(torch.max(preatrained_model['fc1.weight']).item()))
    # print("Mean: " + str(torch.mean(preatrained_model['fc1.weight']).item()))
    #
    # print('\n============== fc2.weight ===================')
    # print("Min: " + str(torch.min(preatrained_model['fc2.weight']).item()))
    # print("Max: " + str(torch.max(preatrained_model['fc2.weight']).item()))
    # print("Mean: " + str(torch.mean(preatrained_model['fc2.weight']).item()))
    #
    # print('\n============== fc3.weight ===================')
    # print("Min: " + str(torch.min(preatrained_model['fc3.weight']).item()))
    # print("Max: " + str(torch.max(preatrained_model['fc3.weight']).item()))
    # print("Mean: " + str(torch.mean(preatrained_model['fc3.weight']).item()))

    model.load_state_dict(torch.load(PATH))
    model.eval()

    test_dataset = create_dataset(path_to_csv=DATASET_PATH,
                                  case="Test",
                                  transform=data_transform(),
                                  channels_vector=CHANNELS)  # create a dataset

    test_dataset_size = len(test_dataset)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Register forward hooks
    # save_output = SaveOutput()
    # hook_handles = []
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.modules.conv.Conv2d):
    #         handle = layer.register_forward_hook(save_output)
    #         hook_handles.append(handle)


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

        # plt.imshow(np.moveaxis(images.cpu()[0].numpy()[0:3, :, :], 0, 2))
        # plt.show()
        #
        # features_list = list(torch.Tensor.cpu(save_output.outputs[0][0]).detach().numpy())
        # display_multiple_img(features_list, 8, 8, scale=20)
        #
        # features_list = list(torch.Tensor.cpu(save_output.outputs[1][0]).detach().numpy())
        # display_multiple_img(features_list, 16, 8, scale=20)
        #
        # features_list = list(torch.Tensor.cpu(save_output.outputs[2][0]).detach().numpy())
        # display_multiple_img(features_list, 16, 16, scale=20)

        # save_output.clear()

        if predicted != labels:  # and predicted.item() == 0:   and img_index <= 40:
            path_to_image = ""
            image = images.cpu()[0].numpy()[0:3, :, :]

            # Save images
            plt.imshow(np.moveaxis(image, 0, 2))
            if predicted.item() == 1:
                plt.savefig('./results/false_positive/fp_' + str(image_index[0]) + '_RGB.png')
            else:
                plt.savefig('./results/false_negative/fn_' + str(image_index[0]) + '_RGB.png')

            # i = 3
            # for channel in ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]:
            #     if predicted.item() == 0:
            #         path_to_image = './results/false_positive/fp_' + str(img_index) + '_' + channel + '.png'
            #     else:
            #         path_to_image = './results/false_negative/fn_' + str(img_index) + '_' + channel + '.png'
            #
            #     image = images.cpu()[0].numpy()[i, :, :]
            #     plt.imshow(image, cmap="gray")  # np.moveaxis(image, 0, 2))
            #     plt.savefig(path_to_image)
            #     i = i + 1

        correct += (predicted == labels).sum().item()
        predicted_arr.append(predicted.item())
        [y_hat.append(output[1].item()) for output in outputs]

    # Confusion Matrix
    plot_confusion_matrix(y, predicted_arr)

    y_hat_ones = []
    y_hat_zeros = []

    [y_hat_ones.append(y_hat[i]) for i, j in enumerate(y) if j == 1.]
    [y_hat_zeros.append(y_hat[i]) for i, j in enumerate(y) if j == 0.]

    plt.hist(y_hat_zeros, color='b', bins=100, label='CLass 0', alpha=0.5)
    plt.hist(y_hat_ones, color='r', bins=100, label='Class 1', alpha=0.5)
    plt.gca().set(title='Freq vs Prob of Class [1]; Test Accuracy: %d %%' %
                        (100 * correct / total), ylabel='Frequency')
    plt.legend()
    plt.show()

    y_hat.clear()
    y.clear()
