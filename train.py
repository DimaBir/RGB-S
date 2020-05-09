import time
import torch
import config
import torch.optim as optim
import matplotlib.pyplot as plt


from torch import nn
from data import create_dataset, data_transform
from models import vgg16_custom, create_model
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

DATASET_PATH = r"D:\Github\RGB-S\data\simulation_2\binary\simulation_dataset.csv"

# TODO: In final version make them as parameters to function. ARGS, for example
RGB_CHANNELS = {
    "num": 3,
    "channels": {
        "R": True, "G": True, "B": True, "S1": False, "S2": False, "S3": False, "S4": False, "S5": False,
        "S6": False, "S7": False
    },
    "model_path": config.RGB_ONLY_3_CHANNELS
}

SPECTRAL_CHANNELS = {
    "num": 7,
    "channels": {
        "R": False, "G": False, "B": False, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
        "S6": True, "S7": True
    },
    "model_path": config.SPECTRAL_ONLY_7_CHANNELS
}

ALL_CHANNELS = {
    "num": 10,
    "channels": {
        "R": True, "G": True, "B": True, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True,
            "S6": True, "S7": True
    },
    "model_path": config.SPECTRAL_RGB_10_CHANNELS
}

DICTIONARY = RGB_CHANNELS

PATH = DICTIONARY["model_path"]
CHANNELS = DICTIONARY["channels"]
NUM_OF_CHANNELS = DICTIONARY["num"]

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Move to parameters, CLEAN THE CODE
    n_epochs = 300
    batch_size = 64
    learning_rate = 0.1

    # Model creation
    model = create_model(num_of_channels=NUM_OF_CHANNELS).to(device)

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

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    print("Model has been loaded successfully")

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # SGD, ADAM, RMSProp

    # factor = decaying factor
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, verbose=True)

    best_validation_accuracy = 0.0
    epoch = 0
    y_hat = []
    y = []

    for epoch in range(epoch, n_epochs):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        train_total = 0
        train_correct = 0
        total_iters = 0  # the total number of training iterations
        running_loss = 0.0
        print("Start of epoch %d / %d" % (epoch + 1, n_epochs))
        for data in train_data_loader:  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += batch_size
            epoch_iter += batch_size

            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
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
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            iter_data_time = time.time()

            # Validate
            correct = 0
            total = 0
            with torch.no_grad():
                for val_data in validation_data_loader:
                    val_images, val_labels = val_data
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

        y_hat_ones = []
        y_hat_zeros = []
        [y_hat_ones.append(y_hat[i]) for i, j in enumerate(y) if j == 1.]
        [y_hat_zeros.append(y_hat[i]) for i, j in enumerate(y) if j == 0.]
        plt.hist(y_hat_zeros, color='b', bins=100, label='CLass 0', alpha=0.5)
        plt.hist(y_hat_ones, color='r', bins=100, label='Class 1', alpha=0.5)
        plt.gca().set(title='Frequency vs Prob of Class [1], epoch: %d Train Acc: %d %%' %
                            (epoch + 1, 100 * train_correct / train_total), ylabel='Frequency')
        # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.legend()
        plt.show()

        print('Training accuracy : %d %%' % (100 * train_correct / train_total))

        y_hat.clear()
        y.clear()

        print("End of the epoch %d / %d \t Time Taken: %d sec" % (
            epoch + 1, n_epochs, time.time() - epoch_start_time))

        scheduler.step(best_validation_accuracy)

    print('Finished Training')
