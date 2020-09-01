import os
import time
import torch
import config
import argparse
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


from torch import nn
from utils import SaveOutput, display_multiple_img, printgradnorm
from data import create_dataset, data_transform
from models import vgg16_custom, create_model, model3d
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


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


def train(train_settings="RGB_CHANNELS"):
    settings = train_settings
    if train_settings == "RGB_CHANNELS":
        settings = config.RGB_CHANNELS
    elif train_settings == "ALL_CHANNELS":
        settings = config.ALL_CHANNELS
    elif train_settings == "ALL_CHANNELS_MULTI":
        settings = config.ALL_CHANNELS_MULTI
    elif train_settings == "RGB_CHANNELS_MULTI":
        settings = config.RGB_CHANNELS_MULTI
    elif train_settings == "SPECTRAL_3_CHANNELS_MULTI":
        settings = config.SPECTRAL_3_CHANNELS_MULTI
    elif train_settings == "SPECTRAL_CHANNELS_MULTI":
        settings = config.SPECTRAL_CHANNELS_MULTI

    dictionary = settings
    print(dictionary["msg"])

    path = dictionary["model_path"]
    channels = dictionary["channels"]
    num_of_channels = dictionary["num"]
    dataset_path = dictionary["dataset_path"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Move to parameters, CLEAN THE CODE
    n_epochs = 600
    batch_size = 96
    learning_rate = 0.1

    # Model creation
    model = create_model(num_of_channels=num_of_channels).to(device)

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

    train_dataset = create_dataset(path_to_csv=dataset_path,
                                   case="Train",
                                   transform=data_transform(),  # create a dataset
                                   channels_vector=channels)

    validation_dataset = create_dataset(path_to_csv=dataset_path,
                                        case="Validation",
                                        transform=data_transform(),  # create a dataset
                                        channels_vector=channels)

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

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())  # , weight_decay=1e-4)  # SGD, ADAM, RMSProp
    #  optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # factor = decaying factor
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, verbose=True)

    best_validation_accuracy = 0.0
    epoch = 0
    y_hat = []
    y = []
    loss_history = []
    train_accuracy = []
    val_accuracy = []

    # TODO: Use Job/TOWATCH/Github Example to refactor the code
    for epoch in range(epoch, n_epochs):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        train_total = 0
        train_correct = 0
        total_iters = 0  # the total number of training iterations
        running_loss = 0.0
        print("Start of epoch %d / %d" % (epoch + 1, n_epochs))
        isFirst = True
        count = 0
        for data in train_data_loader:  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

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

            train_predicted = outputs.argmax(dim=1, keepdim=True)
            train_correct += train_predicted.eq(labels.view_as(train_predicted)).sum().item()

            [y_hat.append(output[1].item()) for output in outputs]

            # L1 regularization
            # l1_regularization = None
            # for W in model.parameters():
            #     l1_regularization = W.norm(p=1)

            loss = criterion(outputs, labels.long())  # + 0.5 * l1_regularization
            loss_history.append(loss)
            # Sanity check
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
            average_val_acc = []
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
            average_val_acc.append(current_validation_acc)

            if current_validation_acc > best_validation_accuracy:
                print('Best validation accuracy : %d %%' % current_validation_acc)
                best_validation_accuracy = current_validation_acc
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                if not os.path.exists(path):
                    open(path, 'w').close()
                torch.save(model.state_dict(), path)
                print("Model saved")

        val_accuracy.append(current_validation_acc)  # sum(average_val_acc) / len(average_val_acc))  # put average val_accuracy

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

        scheduler.step(best_validation_accuracy)

        if epoch % 5 == 0:
            plt.plot(range(len(train_accuracy)), train_accuracy, '#FFA500', label='Training Acc')
            plt.plot(range(len(val_accuracy)), val_accuracy, 'b', label='Validation Acc')
            plt.title('Training and Validation accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

            # plt.plot(range(len(loss_history)), loss_history, 'r', label='Training Loss')
            # plt.title('Training Loss')
            # plt.xlabel('Steps')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.show()

    print('Finished Training')


if __name__ == '__main__':
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=90,
    #                     help='input batch size for training (default: 90)')
    # parser.add_argument('--test-batch-size', type=int, default=1,
    #                     help='input batch size for testing (default: 1)')
    # parser.add_argument('--epochs', type=int, default=600,
    #                     help='number of epochs to train (default: 600)')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='learning rate (default: 0.001)')
    # parser.add_argument('--settings', type=str, default="RGB_CHANNELS", choices=TRAIN_SETTINGS,
    #                     help='Choose type of model to train (default: RGB Channels only)')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()
    TRAIN_SETTINGS = [
        "ALL_CHANNELS",
        "RGB_CHANNELS",
        "ALL_CHANNELS_MULTI",
        "RGB_CHANNELS_MULTI",
        "SPECTRAL_3_CHANNELS_MULTI",
        "SPECTRAL_CHANNELS_MULTI"
    ]
    train("RGB_CHANNELS_MULTI")
