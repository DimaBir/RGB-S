import time
import torch
import config
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


from torch import nn
from utils import SaveOutput, display_multiple_img, printgradnorm
from data import create_dataset, data_transform
from models import vgg16_custom, create_model, model3d
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


# TODO: In final version make them as parameters to function. ARGS, for example
RGB_CHANNELS = {
    "num": 3,
    "channels": {
        "R": True, "G": True, "B": True, "S1": False, "S2": False, "S3": False, "S4": False, "S5": False,
        "S6": False, "S7": False
    },
    "model_path": config.RGB_ONLY_3_CHANNELS,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\binary\simulation_dataset.csv"
}

SPECTRAL_CHANNELS = {
    "num": 7,
    "channels": {
        "R": False, "G": False, "B": False, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
        "S6": True, "S7": True
    },
    "model_path": config.SPECTRAL_ONLY_7_CHANNELS,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\binary\simulation_dataset.csv"
}

ALL_CHANNELS = {
    "num": 10,
    "channels": {
        "R": True, "G": True, "B": True, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
            "S6": True, "S7": True
    },
    "model_path": config.SPECTRAL_RGB_10_CHANNELS,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\binary\simulation_dataset.csv"
}

RGB_CHANNELS_MULTI = {
    "num": 3,
    "channels": {
        "R": True, "G": True, "B": True, "S1": False, "S2": False, "S3": False, "S4": False, "S5": False,
        "S6": False, "S7": False
    },
    "model_path": config.RGB_ONLY_3_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_dataset.csv"
}

SPECTRAL_CHANNELS_MULTI = {
    "num": 7,
    "channels": {
        "R": False, "G": False, "B": False, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
        "S6": True, "S7": True
    },
    "model_path": config.SPECTRAL_ONLY_7_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_spectral_dataset.csv"
}

SHORT_SPECTRAL_CHANNELS_MULTI = {
    "num": 3,
    "channels": {
        "R": False, "G": False, "B": False, "S1": True, "S2": True, "S3": True, "S4": False, "S5": False,
        "S6": False, "S7": False
    },
    "model_path": config.SPECTRAL_ONLY_3_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_spectral_dataset.csv"
}

ALL_CHANNELS_MULTI = {
    "num": 10,
    "channels": {
        "R": True, "G": True, "B": True, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
            "S6": True, "S7": True
    },
    "model_path": config.SPECTRAL_RGB_10_CHANNELS_MULTI,
    "dataset_path": r"D:\Github\RGB-S\data\simulation_3\multi\simulation_dataset.csv"
}

DICTIONARY = SHORT_SPECTRAL_CHANNELS_MULTI

PATH = DICTIONARY["model_path"]
CHANNELS = DICTIONARY["channels"]
NUM_OF_CHANNELS = DICTIONARY["num"]
DATASET_PATH = DICTIONARY["dataset_path"]


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())


def count_mean_and_std_for_dataset(data_loader):
    mean = 0.
    std = 0.
    min = 255.
    max = 0.
    nb_samples = 0.
    for data in data_loader:
        batch_samples = data[0].size(0)
        data = data[0].view(batch_samples, data[0].size(1), -1)

        tmp_min = torch.min(data).item()
        tmp_max = torch.max(data).item()
        min = tmp_min if min > tmp_min else min
        max = tmp_max if max < tmp_max else max

        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("mean: " + str(mean))
    print("std: " + str(std))
    print("min: " + str(min))
    print("max: " + str(max))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Move to parameters, CLEAN THE CODE
    n_epochs = 600
    batch_size = 96
    learning_rate = 0.1

    # Model creation
    model = create_model(num_of_channels=NUM_OF_CHANNELS).to(device)

    # Register forward hooks
    # save_output = SaveOutput()
    # hook_handles = []
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.modules.conv.Conv2d):
    #         handle = layer.register_forward_hook(save_output)
    #         hook_handles.append(handle)
    # model.conv1.register_backward_hook(printgradnorm)
    # model.conv2.register_backward_hook(printgradnorm)
    # model.conv3.register_backward_hook(printgradnorm)

    train_dataset = create_dataset(path_to_csv=DATASET_PATH,
                                   case="Train",
                                   transform=data_transform(),  # create a dataset
                                   channels_vector=CHANNELS)

    validation_dataset = create_dataset(path_to_csv=DATASET_PATH,
                                        case="Validation",
                                        transform=data_transform(),  # create a dataset
                                        channels_vector=CHANNELS)

    train_dataset_size = len(train_dataset)  # get the number of images in the dataset.
    validation_dataset_size = len(validation_dataset)  # get the number of images in the dataset.

    print('The number of training images = %d' % train_dataset_size)
    print('The number of validation images = %d' % validation_dataset_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    # print('Train dataset stat: ')
    # count_mean_and_std_for_dataset(train_data_loader)
    #
    # print('Validation dataset stat: ')
    # count_mean_and_std_for_dataset(validation_data_loader)

    print("Model has been loaded successfully")

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())# , weight_decay=1e-4)  # SGD, ADAM, RMSProp
    #  optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # factor = decaying factor
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=40, verbose=True)

    best_validation_accuracy = 0.0
    epoch = 0
    y_hat = []
    y = []
    train_accuracy = []
    val_accuracy = []

    # TODO: Use Job/TOWATCH/Github Example to refactor the code
    for epoch in range(epoch, n_epochs):  # outer loop for different epochs;
        epoch_start_time = time.time()    # timer for entire epoch
        iter_data_time = time.time()      # timer for data loading per iteration
        epoch_iter = 0                    # the number of training iterations in current epoch, reset to 0 every epoch

        train_total = 0
        train_correct = 0
        total_iters = 0                   # the total number of training iterations
        running_loss = 0.0
        print("Start of epoch %d / %d" % (epoch + 1, n_epochs))
        isFirst = True
        count = 0
        for data in train_data_loader:    # inner loop within one epoch
            iter_start_time = time.time() # timer for computation per iteration

            total_iters += batch_size
            epoch_iter += batch_size

            # Get the inputs; data is a list of [inputs, labels]
            images, labels, _ = data

            [y.append(label.item()) for label in labels]
            images = images.type(torch.cuda.FloatTensor).to(device)
            labels = labels.type(torch.cuda.FloatTensor).to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.predict(images)

            train_total += labels.size(0)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels).sum().item()
            [y_hat.append(output[1].item()) for output in outputs]

            loss = criterion(outputs, labels.long())
            assert (loss == loss)

            print('Loss: ' + str(loss.item()))
            loss.backward()
            optimizer.step()

            # Add Gradient Clipping in order to avoid Gradient Exploding
            # clip_grad_norm_(model.parameters(), 1)

            # Lets see [conv_layer_num], [feature_map_num], [channel_num]
            # if epoch % 10 == 0 and count < 1: # isFirst
            #     features_list = list(torch.Tensor.cpu(save_output.outputs[0][0]).detach().numpy())
            #     display_multiple_img(features_list, 8, 8, scale=20)
            #     count = count + 1
            # save_output.clear()

            # print statistics
            running_loss += loss.item()
            iter_data_time = time.time()

            # Validate
            correct = 0
            total = 0
            with torch.no_grad():
                for val_data in validation_data_loader:
                    val_images, val_labels, _ = val_data
                    val_images = val_images.type(torch.cuda.FloatTensor).to(device)
                    val_labels = val_labels.type(torch.cuda.FloatTensor).to(device)

                    val_outputs = model(val_images)
                    total += val_labels.size(0)
                    _, predicted = torch.max(val_outputs.data, 1)
                    correct += (predicted == val_labels).sum().item()

            current_validation_acc = (100 * correct / total)

            if current_validation_acc > best_validation_accuracy:
                print('Best validation accuracy : %d %%' % current_validation_acc)
                best_validation_accuracy = current_validation_acc
                torch.save(model.state_dict(), PATH)
                print("Model saved")

        val_accuracy.append(current_validation_acc)  # put last accuracy

        y_hat_ones = []
        y_hat_zeros = []
        [y_hat_ones.append(y_hat[i]) for i, j in enumerate(y) if j == 1.]
        [y_hat_zeros.append(y_hat[i]) for i, j in enumerate(y) if j == 0.]
        # plt.hist(y_hat_zeros, color='b', bins=100, label='Class 0', alpha=0.5)
        # plt.hist(y_hat_ones, color='r', bins=100, label='Class 1', alpha=0.5)
        # plt.gca().set(title='Frequency vs Prob of Class [1], epoch: %d Train Acc: %d %%' %
        #                    (epoch + 1, 100 * train_correct / train_total), ylabel='Frequency')

        # plt.legend()
        # plt.show()

        print('Training accuracy : %d %%' % (100 * train_correct / train_total))
        train_accuracy.append(100 * train_correct / train_total)

        y_hat.clear()
        y.clear()

        print("End of the epoch %d / %d \t Time Taken: %d sec" % (
            epoch + 1, n_epochs, time.time() - epoch_start_time))

        # scheduler.step(best_validation_accuracy)

        if epoch % 5 == 1:
            plt.plot(range(len(train_accuracy)), train_accuracy, '#FFA500', label='Training Acc')
            plt.plot(range(len(val_accuracy)), val_accuracy, 'b', label='Validation Acc')
            plt.title('Training and Validation accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

    print('Finished Training')
