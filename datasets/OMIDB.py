import torch
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np

from sklearn.model_selection import train_test_split

from datasets import SubDataset, AbstractDomainInterface

IMG_SIZE = 256

class OMIDB(AbstractDomainInterface):
    """
        OPTIMAM OMI-DB mammographic dataset.
        Variable size depending on filtering
        D1: (80% train + 10% valid) + (10% test)
        D2: 90% valid + 10% test.
    """

    def __init__(self):
        super(OMIDB, self).__init__()

        im_transformer  = transforms.Compose([transforms.ToTensor()])
        size_str = str(IMG_SIZE)
        root_path       = './workspace/datasets/omidb/' + size_str

        base_dataset = datasets.ImageFolder(root_path,transform=im_transformer)

        indices = np.arange(len(base_dataset))
        train_indices_np, test_indices_np = train_test_split(indices, test_size=0.1, stratify=base_dataset.targets)
        
        train_targets = [base_dataset.targets[i] for i in train_indices_np]

        self.D1_train_ind_np, self.D2_valid_ind_np = train_test_split(train_indices_np, test_size=0.1, stratify=train_targets)

        self.D2_valid_ind = torch.from_numpy(train_indices_np) #looks weird but we don't actually train on it!
        self.D2_test_int = torch.from_numpy(test_indices_np)

        self.D1_test_ind = torch.from_numpy(test_indices_np)
        self.D1_train_ind = torch.from_numpy(D1_train_ind_np)
        self.D2_valid_ind = torch.from_numpy(D2_valid_ind_np)

        self.base_dataset = base_dataset

    
    def get_D1_train(self):
        return SubDataset(self.name, self.base_dataset, self.D1_train_ind)
    def get_D1_valid(self):
        return SubDataset(self.name, self.base_dataset, self.D1_valid_ind, label=0)
    def get_D1_test(self):
        return SubDataset(self.name, self.base_dataset, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.base_dataset, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.base_dataset, self.D2_test_ind, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        return transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                   transforms.Grayscale(),
                                   transforms.ToTensor()
                                   ])
