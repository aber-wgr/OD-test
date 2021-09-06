import torch
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data import WeightedRandomSampler

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

        im_transformer  = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.Grayscale(), transforms.ToTensor()])
        size_str = str(IMG_SIZE)
        root_path       = './workspace/datasets/lesion_segments/'

        base_dataset = datasets.ImageFolder(root_path,transform=im_transformer)

        indices = np.arange(len(base_dataset))
        train_indices_np, test_indices_np = train_test_split(indices, test_size=0.1, stratify=base_dataset.targets)
        
        train_targets = [base_dataset.targets[i] for i in train_indices_np]

        D1_train_ind_np, D2_valid_ind_np = train_test_split(train_indices_np, test_size=0.1, stratify=train_targets)

        self.D2_valid_ind = torch.from_numpy(train_indices_np) #looks weird but we don't actually train on it!
        self.D2_test_int = torch.from_numpy(test_indices_np)

        self.D1_test_ind = torch.from_numpy(test_indices_np)
        self.D1_train_ind = torch.from_numpy(D1_train_ind_np)
        self.D2_valid_ind = torch.from_numpy(D2_valid_ind_np)

        self.base_dataset = base_dataset

        self.calculate_D1_weighting()


    def get_weights_by_class(self):
        return self.train_class_weight

#    def calculate_D1_weighting(self):
#        train_set = self.get_D1_train()
#        nc = self.get_num_classes()
#        count = [0] * nc                                                      
#        for item in train_set:                                                         
#            count[item[1]] += 1                                                     
#        self.train_class_weight = [0.] * nc                                      
#        for i in range(nc):                                                   
#            self.train_class_weight[i] = 1.0/float(count[i])

    
    def get_D1_train(self):
        return SubDataset(self.name, self.base_dataset, self.D1_train_ind)
    def get_D1_valid(self):
        return SubDataset(self.name, self.base_dataset, self.D1_valid_ind, label=0)
    def get_D1_test(self):
        return SubDataset(self.name, self.base_dataset, self.D1_test_ind, label=0)

    def get_D1_train_weighting(self):
        d1_set = self.get_D1_train()
        weights = [0] * len(d1_set)                                              
        for idx, val in enumerate(d1_set):                                          
            weights[idx] = self.train_class_weight[val[1]]                                  
        return weights

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.base_dataset, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.base_dataset, self.D2_test_ind, label=1, transform=D1.conformity_transform())

    def get_num_classes(self):
        return 2

    def get_train_sampler(self):
        return WeightedRandomSampler(self.get_D1_train_weighting(), len(self.D1_train_ind),replacement=False)

    def conformity_transform(self):
        return transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                   transforms.Grayscale(),
                                   transforms.ToTensor()
                                   ])
