from __future__ import print_function
import os
import os.path
import errno
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from datasets import SubDataset, AbstractDomainInterface

class GeneratedDataset(data.Dataset):
    """`Dataset used for generated data
    This was originally built for the domain shift experiment, but can be used for any generated data.
    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = True  # training set or test set

        # self.train_data contains the images and targets as created by the generator
        # self.train_labels contains the labels for the images

        self.train_data = []
        self.train_labels = []

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)

    def _check_exists(self):
        return len(self.train_data) > 0

    def append(self, data, label):
        self.train_data.append(data)
        self.train_labels.append(label)

class GeneratedDomain(AbstractDomainInterface):

    def __init__(self, base_array=None, drop_class=None):
        super(GeneratedDomain, self).__init__(drop_class = drop_class)
        
        im_transformer  = transforms.Compose([transforms.ToTensor()])
        self.ds_train   = GeneratedDataset(transform=im_transformer)
        if base_array is not None:
            for i in range(len(base_array)):
                self.ds_train.append(base_array[i], i)
    
    def get_D2_train(self, D1):
        return self.ds_train

    def get_D2_valid(self, D1):
        return self.ds_train

    def get_D2_test(self, D1):
        return self.ds_train

    def get_num_classes(self):
        return len(torch.unique(self.ds_train))
