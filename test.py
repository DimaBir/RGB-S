import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models import create_model
from torch.utils.data import DataLoader
from data import create_dataset, data_transform
from utils import occlusion

PATH = "rgb_s_cnn_with_spectral_10.pt"

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    test_dataset = create_dataset(path_to_csv=".//data//simulation_2//binary//simulation_dataset.csv",
                                  case="Test",
                                  transform=data_transform())  # create a dataset

    test_dataset_size = len(test_dataset)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Test
    correct = 0
    total = 0
    y_hat = []
    y = []

    for data in test_data_loader:
        images, labels = data
        [y.append(label.item()) for label in labels]
        images = images.type(torch.cuda.FloatTensor).to(device)
        labels = labels.type(torch.cuda.FloatTensor).to(device)
        # heatmap = occlusion(model, images, labels[0].item(), 1, 1)

        # imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=1.)
        # figure = imgplot.get_figure()
        # figure.show()
        outputs = model.predict(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        if predicted == labels and predicted.item() == 0:
            plt.imshow(np.moveaxis(images.cpu()[0].numpy()[0:1, :, :], 0, 2).reshape((128, 128)), cmap='gray')
            plt.show()

        correct += (predicted == labels).sum().item()
        [y_hat.append(output[1].item()) for output in outputs]

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