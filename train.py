import sys
import time
import torch
import shutil
import torch.optim as optim

from torch import nn
from data import create_dataset
from models import vgg16_custom
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

_, term_width = shutil.get_terminal_size()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time


def progress_bar(current, totals, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / totals)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, totals))

    if current < totals - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = 1
    n_epochs = 100
    batch_size = 32

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = create_dataset(path_to_csv=".//data//simulation_binary_1//simulation_dataset.csv",
                                   case="Train",
                                   transform=data_transform)  # create a dataset
    validation_dataset = create_dataset(path_to_csv=".//data//simulation_binary_1//simulation_dataset.csv",
                                        case="Validation",
                                        transform=data_transform)  # create a dataset
    train_dataset_size = len(train_dataset)  # get the number of images in the dataset.
    validation_dataset_size = len(validation_dataset)  # get the number of images in the dataset.

    print('The number of training images = %d' % train_dataset_size)
    print('The number of validation images = %d' % validation_dataset_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    model = vgg16_custom.create_model().to(device)  # create a model

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate) # SGD, ADAM, RMSProp

    # factor = decaying factor
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    best_validation_accuracy = 0.0
    epoch = 0

    for epoch in range(epoch, n_epochs):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        total_iters = 0  # the total number of training iterations
        running_loss = 0.0
        print("Start of epoch %d / %d" % (epoch + 1, n_epochs))
        for data in train_data_loader:  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % 25 == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += batch_size
            epoch_iter += batch_size

            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images = images.type(torch.cuda.FloatTensor).to(device)
            labels = labels.type(torch.cuda.FloatTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if total_iters % 25 == 0:  # print every 100 mini-batches
                print('[%d/%d, %5d/%d] loss: %.3f' %
                      (epoch + 1, n_epochs, total_iters + 1, len(train_data_loader), running_loss / 25))
                running_loss = 0.0

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
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            current_validation_acc = (100 * correct / total)

            # progress_bar(total_iters, len(train_data_loader), 'Loss: %.3f '
            #             % (current_validation_acc))
            if current_validation_acc > best_validation_accuracy:
                print('Best validation accuracy : %d %%' % current_validation_acc)
                best_validation_accuracy = current_validation_acc
                torch.save(model.state_dict(), "rgb_s_cnn.pt")
                print("Model saved")

        print("End of epoch %d / %d \t Time Taken: %d sec" % (
            epoch, n_epochs, time.time() - epoch_start_time))

        scheduler.step(best_validation_accuracy)

    print('Finished Training')
