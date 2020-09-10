import os
import time
import torch
import config
import argparse
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch import nn
from utils import SaveOutput, display_multiple_img, printgradnorm, ChannelSettings
from data import create_dataset, data_transform
from models import create_model
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from utils import count_mean_and_std_for_dataset, printnorm


def train(n_epochs=600, learning_rate=0.1, batch_size=96, FC_dp=0.0, train_settings=None,
          show_feature_maps=False, print_dataset_mean_std=False, use_scheduler=False, use_L1_regularization=False,
          show_accuracy=True):
    DICTIONARY = config.SETTINGS[train_settings]
    print(DICTIONARY.msg)

    PATH = DICTIONARY.model_path
    CHANNELS = DICTIONARY.channels
    NUM_OF_CHANNELS = DICTIONARY.num
    DATASET_PATH = DICTIONARY.dataset_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model creation
    model = create_model(num_of_channels=NUM_OF_CHANNELS, FC_dp=FC_dp).to(device)

    if show_feature_maps:
        # Register forward hooks
        save_output = SaveOutput()
        hook_handles = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)
        model.conv1.register_backward_hook(printgradnorm)
        model.conv2.register_backward_hook(printgradnorm)
        model.conv3.register_backward_hook(printgradnorm)

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

    if print_dataset_mean_std:
        print(f'Train dataset stat: {count_mean_and_std_for_dataset(train_data_loader)}')
        print(f'Validation dataset stat: {count_mean_and_std_for_dataset(validation_data_loader)}')

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    # factor = decaying factor
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, verbose=True)

    best_validation_accuracy = 0.0
    epoch = 0

    loss_history = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(epoch, n_epochs):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        train_total = 0
        train_correct = 0
        total_iters = 0  # the total number of training iterations
        running_loss = 0.0

        print("Start of epoch %d / %d" % (epoch + 1, n_epochs))

        count = 0
        for data in train_data_loader:  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += batch_size
            epoch_iter += batch_size

            # Get the inputs; data is a list of [inputs, labels]
            images, labels, _ = data

            images = images.type(torch.cuda.FloatTensor).to(device)
            labels = labels.type(torch.cuda.FloatTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.predict(images)

            train_total += labels.size(0)

            train_predicted = outputs.argmax(dim=1, keepdim=True)
            train_correct += train_predicted.eq(labels.view_as(train_predicted)).sum().item()

            if use_L1_regularization:
                # L1 regularization
                l1_regularization = None
                for W in model.parameters():
                    l1_regularization = W.norm(p=1)

            loss = criterion(outputs, labels.long())  # + 0.5 * l1_regularization
            loss_history.append(loss)

            # Sanity check
            assert (loss == loss)
            print('Loss: ' + str(loss.item()))

            loss.backward()
            optimizer.step()

            if show_feature_maps:
                # Lets see [conv_layer_num], [feature_map_num], [channel_num]
                if epoch % 10 == 0 and count < 1:  # isFirst
                    features_list = list(torch.Tensor.cpu(save_output.outputs[0][0]).detach().numpy())
                    display_multiple_img(features_list, 8, 8, scale=20)
                    count = count + 1
                save_output.clear()

            # Print statistics
            running_loss += loss.item()

            # Validate
            total = 0
            correct = 0

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

                if not os.path.exists(os.path.dirname(PATH)):
                    os.makedirs(os.path.dirname(PATH))
                if not os.path.exists(PATH):
                    open(PATH, 'w').close()

                torch.save(model.state_dict(), PATH)
                print("Model saved")

        val_accuracy.append(current_validation_acc)

        print('Training accuracy : %d %%' % (100 * train_correct / train_total))
        train_accuracy.append(100 * train_correct / train_total)
        print("End of the epoch %d / %d \t Time Taken: %d sec" % (
            epoch + 1, n_epochs, time.time() - epoch_start_time))

        if use_scheduler:
            scheduler.step(best_validation_accuracy)

        if show_accuracy:
            if epoch % 5 == 0:
                plt.plot(range(len(train_accuracy)), train_accuracy, '#FFA500', label='Training Acc')
                plt.plot(range(len(val_accuracy)), val_accuracy, 'b', label='Validation Acc')
                plt.title('Training and Validation accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

    print('Finished Training')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Phenomics RGB-T Classifier')
    parser.add_argument('--mean_std', type=bool, default=False,
                        help='print dataset mean std (default: False)')
    parser.add_argument('--scheduler', type=bool, default=False,
                        help='Use learning rae decay technique (default: False)')
    parser.add_argument('--L1', type=bool, default=False,
                        help='Use L1 regularization (default: False)')
    parser.add_argument('--batch-size', type=int, default=96,
                        help='input batch size for training (default: 96)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=600,
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--fc', type=float, default=0.0,
                        help='First Fully Connected layer dropout (default: 0.0)')
    parser.add_argument('--settings', type=str, choices=config.CHOICES, help='Choose type of model to train')
    args = parser.parse_args()

    train(n_epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size, FC_dp=0.0,
          train_settings=args.settings, show_feature_maps=False,
          print_dataset_mean_std=False, use_scheduler=False, use_L1_regularization=False, show_accuracy=True)
