__author__ = 'KSM'

"""
Author: Kunjan Mhaske

This program helps data.py program to fetch the images from the respective dataset
directory to pre-process / transform them for training and testing purpose.

"""

import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image

def is_image_file(filename):
    """
    This method is used to check if the given file is an image file or not
    :param filename: image filename
    :return: true if it is image else false
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    """
    This method is used to load the particular image in YCbCr format
    :param filepath: image filepath
    :return: Y part of image matrix
    """
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class DatasetFromFolder(data.Dataset):
    """
    This class is used to get perform the transform operations on the images of dataset
    """
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        """
        This applies the transformation operations on the image dataset
        :param image_dir:
        :param input_transform:
        :param target_transform:
        """
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        This method is used to fetch the images from the given path and applies the transformation
        on input or target images accordingly
        :param index: filepath
        :return: input image and target image
        """
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        """
        This method returns the length of the given dataset images
        :return: length of image dataset
        """
        return len(self.image_filenames)
