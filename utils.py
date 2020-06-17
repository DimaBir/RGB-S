# custom function to conduct occlusion experiments
import torch
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # image = pad_if_smaller(image, self.size)
        crop_params = T.RandomCrop.get_params(np.array(image), (self.size, self.size))
        image = F.crop(image, *crop_params)

        return image

def plot_confusion_matrix(actual_labels, predicted_labels):
    array = confusion_matrix(actual_labels, predicted_labels)
    df_cm = pd.DataFrame(array, index=[i for i in ["GT: Healthy", "GT: Ill"]],
                         columns=[i for i in ["PR: Healthy", "PR: Ill"]])

    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='d')  # font size

    plt.show()


def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    """
    :param model:
    :param image:
    :param label:
    :param occ_size:
    :param occ_stride:
    :param occ_pixel:
    :return:

    Example:
        heatmap = occlusion(model, images, pred[0].item(), 32, 14)

        # displaying the image using seaborn heatmap and also setting the maximum value of gradient to probability
        imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=prob_no_occ)
        figure = imgplot.get_figure()
    """
    # Get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # Setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # Create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # Iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            # Replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # Run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][1]

            # Setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap
