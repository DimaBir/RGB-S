import argparse
import os
import PIL
import random
import shutil
import numpy as np
import pandas as pd

from PIL import Image


def extractandsaveRGBchannels(image_path, channel_to_extract='r') -> str:
    channel = {'r': 0, 'g': 1, 'b': 2}

    # Read RGB image from the path and extract specific channel
    rgb_image_as_array = np.array(PIL.Image.open(image_path))[:, :, channel[channel_to_extract]]
    image = Image.fromarray(rgb_image_as_array)

    # Add channel suffix to extracted channel image path
    new_filename = os.path.splitext(image_path)[0] + "_" + channel_to_extract + ".png"
    image.save(new_filename)

    return new_filename


def generate(root_dir, type, log_file="simulation_log.txt") -> pd.DataFrame():
    """
    root_dir: .//simulation_binary_1/
    log_file: simulation_log.txt
    image_folder: root_dir + 'images/'
    :param log_file:
    :type type: object
    """

    current_directory = os.getcwd()
    image_folder = os.path.join(current_directory, root_dir, 'images//')
    labels_file = os.path.join(root_dir, log_file)

    # Open the label file file
    file = open(labels_file, "r")

    # Loop through lines and split to images and labels
    images_and_labels = [tuple(line.split()) for line in file]

    # Create DataFrame
    df = pd.DataFrame()
    df['image_path'] = [os.path.join(image_folder, image_and_label[0]) for image_and_label in images_and_labels]

    # Extract RGB Image Channels to different images and save path
    rgb_images = [os.path.join(image_folder, image_and_label[0] + '_00.png') for image_and_label in images_and_labels]

    df['image_R'] = [extractandsaveRGBchannels(image, 'r') for image in rgb_images]
    df['image_G'] = [extractandsaveRGBchannels(image, 'g') for image in rgb_images]
    df['image_B'] = [extractandsaveRGBchannels(image, 'b') for image in rgb_images]

    # Safe path to each spectral channel image
    df['image_S1'] = [os.path.join(image_folder, image_and_label[0] + '_01.png') for image_and_label in
                      images_and_labels]
    df['image_S2'] = [os.path.join(image_folder, image_and_label[0] + '_02.png') for image_and_label in
                      images_and_labels]
    df['image_S3'] = [os.path.join(image_folder, image_and_label[0] + '_03.png') for image_and_label in
                      images_and_labels]
    df['image_S4'] = [os.path.join(image_folder, image_and_label[0] + '_04.png') for image_and_label in
                      images_and_labels]
    df['image_S5'] = [os.path.join(image_folder, image_and_label[0] + '_05.png') for image_and_label in
                      images_and_labels]
    df['image_S6'] = [os.path.join(image_folder, image_and_label[0] + '_06.png') for image_and_label in
                      images_and_labels]
    df['image_S7'] = [os.path.join(image_folder, image_and_label[0] + '_07.png') for image_and_label in
                      images_and_labels]

    # Labels get 1 if at least one 1 exists, else 0
    df['label'] = ['1' if '1' in list(image_and_label[1:]) else '0' for image_and_label in images_and_labels]

    case_column = []
    [case_column.insert(i, type) for i in range(len(df))]

    df['case'] = case_column
    return df


def moveto(sourcepath, destinationpath):
    print(f"--- Moving images to {destinationpath} ---")
    if not os.path.exists(destinationpath):
        os.makedirs(destinationpath)
    sourcefiles = os.listdir(sourcepath)
    for file in sourcefiles:
        if file.endswith('.png'):
            shutil.move(os.path.join(sourcepath, file), os.path.join(destinationpath, file))


def generate_partioned_csv(dataset_relative_path, csv_name="simulation_dataset.csv"):
    csv_absolute_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_relative_path)
    csv_absolute_filename = os.path.join(csv_absolute_folder, csv_name)

    print("--- Creating test dataset ---")
    type = 'test'
    folder = os.path.join(csv_absolute_folder, type)

    moveto(os.path.join(csv_absolute_folder, type), os.path.join(folder, 'images'))
    df_test = generate(os.path.join(dataset_relative_path, type), type=type)

    print("--- Creating train dataset ---")
    type = 'train'
    folder = os.path.join(csv_absolute_folder, type)

    moveto(os.path.join(csv_absolute_folder, type), os.path.join(folder, 'images'))
    df_train = generate(os.path.join(dataset_relative_path, type), type=type)

    print("--- Creating validation dataset ---")
    type = 'validation'
    folder = os.path.join(csv_absolute_folder, type)

    moveto(os.path.join(csv_absolute_folder, type), os.path.join(folder, 'images'))
    df_validation = generate(os.path.join(dataset_relative_path, type), type=type)

    print("--- Writing the dataset to csv ---")

    df = pd.concat([df_train, df_validation, df_test], axis=0)
    df.to_csv(csv_absolute_filename, index=False)

    print(f"--- Dataset is ready in path: {csv_absolute_filename} ---")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--path', type=str, help='Relative path to dataset (For example: simulation_5_affine\\binary)')
    args = parser.parse_args()

    generate_partioned_csv(dataset_relative_path=args.path)