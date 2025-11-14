from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint
from utils import augment

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'unlabeled', 'val', 'test']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            # Remove .DS_Store file
            if it_im == '.DS_Store':
                continue
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)

    elif mode == 'unlabeled':
        unlabeled_img_path = os.path.join(root, 'train', 'Img-Unlabeled')

        images = os.listdir(unlabeled_img_path)

        images.sort()

        for it_im in images:
            # Remove .DS_Store file
            if it_im == '.DS_Store':
                continue
            item = (os.path.join(unlabeled_img_path, it_im), None) # No mask
            items.append(item)

    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            # Remove .DS_Store file
            if it_im == '.DS_Store':
                continue
            item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)
    
    else:
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Dataset for Medical Images."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False, data_list=None):
        """
        Args:
            mode (string): One of 'train', 'unlabeled', 'val', 'test', or 'pseudo'.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
            augment (bool): Whether to apply data augmentation.
            equalize (bool): Whether to apply histogram equalization.
            data_list (list): Optional. If provided, use this list of data instead of reading from files.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode

        if data_list is not None:
            self.imgs = data_list
        else:
            self.imgs = make_dataset(root_dir, mode)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        item = self.imgs[index]

        if len(item) == 2:
            # Data from files
            img_path, mask_path = item
            img = Image.open(img_path).convert('L')  # Ensure grayscale
            if mask_path is not None:
                mask = Image.open(mask_path).convert('L')
            else:
                # For unlabeled data, create a dummy mask with the same size as img
                width, height = img.size  # PIL Image size: (width, height)
                mask = Image.new('L', (width, height), 0)  # Create a black mask (all zeros)
            img_name = img_path

        elif len(item) == 3:
            # In-memory data
            img, mask, img_name = item

            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            if isinstance(mask, torch.Tensor):
                mask = transforms.ToPILImage()(mask)
        else:
            raise ValueError("Invalid item format in dataset.")

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.augmentation and (self.mode != 'unlabeled'):
            img, mask, _ = augment(img, mask)

        return [img, mask, img_name]

    # Remove items when the model is trained with pseudo-labeling
    def remove_items(self, img_paths_to_remove):
        # Remove items with specified image paths
        self.imgs = [item for item in self.imgs if item[0] not in img_paths_to_remove]
