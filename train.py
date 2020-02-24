import time

import torch
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data import create_dataset
from models import create_model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = 0.0002
    n_epochs = 2
    batch_size = 1

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = create_dataset(path_to_csv=".//data//simulation_binary_1//simulation_dataset.csv",
                                   case="Train",
                                   transform=data_transform)  # create a dataset
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = create_model().to(device)  # create a model

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    total_iters = 0  # the total number of training iterations
    epoch = 0

    for epoch in range(epoch, n_epochs):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        running_loss = 0.0
        #for i in range(len(train_dataset)):
        for data in data_loader:  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % 10 == 0:
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
            if total_iters % 10:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, total_iters + 1, running_loss / 10))
                running_loss = 0.0

            iter_data_time = time.time()
        #if epoch % 2 == 0:  # cache our model every <save_epoch_freq> epochs
        #    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #    model.save_networks(".//", optimizer, epoch)

        print("\rEnd of epoch %d / %d \t Time Taken: %d sec" % (
            epoch, n_epochs, time.time() - epoch_start_time), end='')
        # model.update_learning_rate()  # update learning rates at the end of every epoch.

    print('Finished Training')