__author__ = 'KSM'

"""
Author: Kunjan Mhaske

This program downloads the BSDS300 / BSDS500 dataset and extract it to the proper folder 
for further process. The program further transforms these images for testing and training
separately. It is used in generate_model.py to get a testing and a training set of images.
"""

from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from dataset import DatasetFromFolder

def download_bsd(dest="dataset"):
    """
    This method is used to download the bsds dataset and extract the images from it to folder.
    :param dest: destination folder name
    :return: path of saved dataset
    """
    # output_image_dir = join(dest, "BSDS500/images")
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        # url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz" # for BSDS500 dataset
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"   # for BSDS300 dataset
        print("downloading url ", url)
        data = urllib.request.urlopen(url)
        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)
        remove(file_path)
    return output_image_dir

def calculate_valid_crop_size(crop_size, upscale_factor):
    """
    This method calculates the proper crop size according to upscale factor given by
    user.
    :param crop_size: crop size
    :param upscale_factor: upscale factor
    :return: proper crop size
    """
    return crop_size - (crop_size % upscale_factor)

def input_transform(crop_size, upscale_factor):
    """
    This method transforms the input images according to upscale factor and proper
    crop size.
    :param crop_size: proper crop size
    :param upscale_factor: upscale factor
    :return: composition of proper transformed image in the form of tensor
    """
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])

def target_transform(crop_size):
    """
    This method is used to transform the target image according to crop size
    :param crop_size: proper crop size
    :return: compisition of proper transformed image in the form of tensor
    """
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def get_training_set(upscale_factor):
    """
    This method is used to get the proper set of images for training the model
    :param upscale_factor: upscale factor
    :return: generated dataset
    """
    root_dir = download_bsd()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

def get_test_set(upscale_factor):
    """
    This method is used to get the proper set of images for testing the model
    :param upscale_factor: upscale factor
    :return: generated dataset
    """
    root_dir = download_bsd()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))