import os
import sys
import PIL
import random
import numpy as np
import pandas as pd

from PIL import Image


def extractandsaveRGBchannels(image_path, channel_to_extract='r'):
    channel = -1

    if channel_to_extract == 'r':
        channel = 0
    elif channel_to_extract == 'g':
        channel = 1
    elif channel_to_extract == 'b':
        channel = 2

    # Read RGB image from the path and extract specific channel
    rgb_image_as_array = np.array(PIL.Image.open(image_path))[:, :, channel]
    image = Image.fromarray(rgb_image_as_array)

    # Add channel suffix to extracted channel image path
    new_filename = os.path.splitext(image_path)[0] + "_" + channel_to_extract + ".png"
    image.save(new_filename)

    return new_filename


def createcsvfile(root_dir, log_file, csv_file_name):
    """
    root_dir: .//simulation_binary_1/
    log_file: simulation_log.txt
    image_folder: root_dir + 'images/'
    """

    current_directory = os.getcwd()
    image_folder = os.path.join(current_directory, root_dir, 'images//')
    labels_file = os.path.join(root_dir, log_file)

    # Open the label file
    file = open(labels_file, "r")

    # Loop through lines in and split to images and labels
    images_and_labels = [tuple(line.split()) for line in file]

    # Create DataFrame
    df = pd.DataFrame()
    df['image_path'] = [os.path.join(image_folder, image_and_label[0]) for image_and_label in images_and_labels]

    # Safe path to each spectral channel image
    df['image_S1'] = [os.path.join(image_folder, image_and_label[0] + '_01.png') for image_and_label in
                      images_and_labels]
    df['image_S2'] = [os.path.join(image_folder, image_and_label[0] + '_02.png') for image_and_label in
                      images_and_labels]
    df['image_S3'] = [os.path.join(image_folder, image_and_label[0] + '_03.png') for image_and_label in
                      images_and_labels]
    # df['image_S4'] = [os.path.join(image_folder, image_and_label[0] + '_04.png') for image_and_label in
    #                   images_and_labels]
    # df['image_S5'] = [os.path.join(image_folder, image_and_label[0] + '_05.png') for image_and_label in
    #                   images_and_labels]
    # df['image_S6'] = [os.path.join(image_folder, image_and_label[0] + '_06.png') for image_and_label in
    #                   images_and_labels]
    # df['image_S7'] = [os.path.join(image_folder, image_and_label[0] + '_07.png') for image_and_label in
    #                   images_and_labels]

    train_indexes = random.sample(range(0, len(df.index)), int(0.8 * len(df.index)))
    validation_test_indexes = [i for i in range(0, len(df.index)) if i not in train_indexes]
    validation_indexes = [validation_test_indexes[i] for i in range(0, len(validation_test_indexes)) if i % 2 == 0]
    test_indexes = [validation_test_indexes[i] for i in range(0, len(validation_test_indexes)) if i % 2 != 0]

    # Labels
    df['label'] = ['1' if '1' in list(image_and_label[2:5]) else '0' for image_and_label in images_and_labels]

    case_column = []
    [case_column.insert(i, 'Train') for i in train_indexes]
    [case_column.insert(i, 'Test') for i in test_indexes]
    [case_column.insert(i, 'Validation') for i in validation_indexes]

    df['case'] = case_column

    df.to_csv(os.path.join(root_dir, csv_file_name), index=False)


if __name__ == '__main__':
    createcsvfile("simulation_3\\multi\\", "simulation_log.txt", csv_file_name='simulation_spectral_short_dataset.csv')
