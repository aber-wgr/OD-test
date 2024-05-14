"""
    This file lists all the global variables that are used throughout the project.
    The two major components of this file are the list of the datasets and the list of the models.
"""

"""
    This is where we keep a reference to all the dataset classes in the project.
"""
import datasets.MNIST as MNIST
import datasets.FashionMNIST as FMNIST
import datasets.notMNIST as NMNIST
import datasets.CIFAR as CIFAR
import datasets.noise as noise
import datasets.STL as STL
import datasets.TinyImagenet as TI
import datasets.OMIDB as OMIDB

import torch
import copy
import utils.distributed as distrib 

class DatasetStore(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.all_dataset_classes = [ MNIST.MNIST, FMNIST.FashionMNIST, NMNIST.NotMNIST,
                        CIFAR.CIFAR10, CIFAR.CIFAR100,
                        STL.STL10, TI.TinyImagenet,
                        noise.UniformNoise, noise.NormalNoise,
                        STL.STL10d32, TI.TinyImagenetd32, OMIDB.OMIDB]
        
        """
        Not all the datasets can be used as a Dv, Dt (aka D2) for each dataset.
        The list below specifies which datasets can be used as the D2 for the other datasets.
        For instance, STL10 and CIFAR10 cannot face each other because they have 9 out 10 classes
        in common.
        """

        self.d2_compatiblity = {
            # This can be used as d2 for            # this
            'MNIST'                                 : ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
            'NotMNIST'                              : ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
            'FashionMNIST'                          : ['MNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
            'CIFAR10'                               : ['MNIST', 'FashionMNIST', 'CIFAR100', 'TinyImagenet', 'TinyImagenetd32'],
            'CIFAR100'                              : ['MNIST', 'FashionMNIST', 'CIFAR10', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
            'STL10'                                 : ['MNIST', 'FashionMNIST', 'CIFAR100', 'TinyImagenet', 'TinyImagenetd32'],
            'TinyImagenet'                          : ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'STL10d32'],
            'OMIDB'                                 : ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32']
        # STL10 is not compatible with CIFAR10 because of the 9-overlapping classes.
        # Erring on the side of caution.
        }

        # We can augment the following training data with mirroring.
        # We make sure there's no information leak in-between tasks.
        self.mirror_augment = {
            'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'
        }

        self.dataset_scales = {
            'MNIST':                  '1,28,28',
            'FashionMNIST':           '1,28,28',
            'CIFAR10':                '3,32,32',
            'CIFAR100':               '3,32,32',
            'STL10':                  '3, 96, 96',
            'TinyImagenet':           '3, 64, 64',
            'OMIDB':                  '1,256,256'
            }

    def get_dataset_scale(self,dataset):
        scale_string = self.dataset_scale[dataset]
        scale = tuple(map(int, scale_string.split(', ')))
        return scale
    
    def generate(self,args):
        self.all_datasets = {}
        for dscls in self.all_dataset_classes:
            self.all_datasets[dscls.__name__] = dscls

import models.classifiers as CLS
import models.autoencoders as AES
import models.pixelcnn.model as PCNN

class ModelFactory(object):
    def __init__(self, parent_class, **kwargs):
        self.parent_class = parent_class
        self.kwargs = kwargs

    def add(self,arg,value):
        self.kwargs[arg] = value

    def __call__(self):
        return self.parent_class(**self.kwargs)
    
class ModelStore(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self,args):
        # if we're using distributed training, we need to know how many GPUs are available to this process
        devices_available = len(args.gpulist)

        if(devices_available > 1):
            VGG_modeltype = CLS.Scaled_VGG_2GPU_Pipeline
            ResNet_modeltype = CLS.Scaled_Resnet_2GPU_Pipeline
        else:
            VGG_modeltype = CLS.Scaled_VGG
            ResNet_modeltype = CLS.Scaled_Resnet

        """
            Each dataset has a list of compatible neural netwok architectures.
            Your life would be simpler if you keep the same family as the same index within each dataset.
            For instance, VGGs are all 0 and Resnets are all 1.
        """

        self.dataset_reference_classifiers = {
            'MNIST':                  [ModelFactory(VGG_modeltype, scale=(1,28,28), classes=10, epochs=60), ModelFactory(ResNet_modeltype, scale=(1,28,28), classes=10, epochs=60), ModelFactory(CLS.Scaled_ResNext, scale=(1,28,28), classes=10, epochs=60)],
            'FashionMNIST':           [ModelFactory(VGG_modeltype, scale=(1,28,28), classes=10, epochs=60), ModelFactory(ResNet_modeltype, scale=(1,28,28), classes=10, epochs=60), ModelFactory(CLS.Scaled_ResNext, scale=(1,28,28), classes=10, epochs=60)],
            'CIFAR10':                [ModelFactory(VGG_modeltype, scale=(3,32,32), classes=10, epochs=60), ModelFactory(ResNet_modeltype, scale=(3,32,32), classes=10, epochs=60), ModelFactory(CLS.Scaled_ResNext, scale=(3,32,32), classes=10, epochs=60)],
            'CIFAR100':               [ModelFactory(VGG_modeltype, scale=(3,32,32), classes=100, epochs=60), ModelFactory(ResNet_modeltype, scale=(3,32,32), classes=100, epochs=60), ModelFactory(CLS.Scaled_ResNext, scale=(3,32,32), classes=100, epochs=60)],
            'STL10':                  [ModelFactory(VGG_modeltype, scale=(3, 96, 96), classes=10, epochs=60), ModelFactory(ResNet_modeltype, scale=(3, 96, 96), classes=10, epochs=60), ModelFactory(CLS.Scaled_ResNext, scale=(3, 96, 96), classes=10, epochs=60)],
            'TinyImagenet':           [ModelFactory(VGG_modeltype, scale=(3, 64, 64), classes=200, epochs=60), ModelFactory(ResNet_modeltype, scale=(3, 64, 64), classes=200, epochs=60), ModelFactory(CLS.Scaled_ResNext, scale=(3, 64, 64), classes=200, epochs=60)],
            #'OMIDB':                  [ModelFactory(VGG_modeltype, scale=(1, 256, 256), classes=5, epochs=80), ModelFactory(ResNet_modeltype, scale=(1, 256, 256), classes=5, epochs=80), ModelFactory(CLS.Scaled_ResNext, scale=(1, 256, 256), classes=5, epochs=80)],
        }

        self.dataset_reference_autoencoders = {
            'MNIST':              [ModelFactory(AES.Generic_AE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
            'FashionMNIST':       [ModelFactory(AES.Generic_AE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
            'CIFAR10':            [ModelFactory(AES.Generic_AE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
            'CIFAR100':           [ModelFactory(AES.Generic_AE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
            'STL10':              [ModelFactory(AES.Generic_AE, dims=(3, 96, 96), max_channels=512, depth=12, n_hidden=512)],
            'TinyImagenet':       [ModelFactory(AES.Generic_AE, dims=(3, 64, 64), max_channels=512, depth=12, n_hidden=512)],
            #'OMIDB':              [ModelFactory(AES.Generic_AE, dims=(1, 256, 256), max_channels=512, depth=12, n_hidden=512)],
        }

        self.dataset_reference_vaes = {
            'MNIST':              [ModelFactory(AES.Generic_VAE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
            'FashionMNIST':       [ModelFactory(AES.Generic_VAE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
            'CIFAR10':            [ModelFactory(AES.Generic_VAE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
            'CIFAR100':           [ModelFactory(AES.Generic_VAE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
            'STL10':              [ModelFactory(AES.Generic_VAE, dims=(3, 96, 96), max_channels=512, depth=12, n_hidden=512)],
            'TinyImagenet':       [ModelFactory(AES.Generic_VAE, dims=(3, 64, 64), max_channels=512, depth=12, n_hidden=512)],
            #'OMIDB':              [ModelFactory(AES.Generic_VAE, dims=(1, 256, 256), max_channels=512, depth=12, n_hidden=512)],
        }

    def get_ref_classifier(self,dataset):
        if dataset in self.dataset_reference_classifiers:
            return self.dataset_reference_classifiers[dataset]
        raise NotImplementedError()
    
    def get_ref_autoencoder(self,dataset):
        if dataset in self.dataset_reference_autoencoders:
            return self.dataset_reference_autoencoders[dataset]
        raise NotImplementedError()
    
    def get_ref_vae(self,dataset):
        if dataset in self.dataset_reference_vaes:
            return self.dataset_reference_vaes[dataset]
        raise NotImplementedError()
    

"""
    This is where we keep a reference to all the methods.
"""

import methods.base_threshold as BT
import methods.score_svm as SSVM
import methods.logistic_threshold as KL
import methods.mcdropout as MCD
import methods.nearest_neighbor as KNN
import methods.binary_classifier as BinClass
import methods.deep_ensemble as DE
import methods.odin as ODIN
import methods.reconstruction_error as RE
import methods.pixelcnn as PCNN
import methods.openmax as OM
import methods.sd_threshold as SD

class MethodStore(object):
    def __init__(self):
        self.all_methods = {
            'prob_threshold':   BT.ProbabilityThreshold,
            'score_svm':        SSVM.ScoreSVM,
            'logistic_svm':     KL.LogisticSVM,
            'mcdropout':        MCD.MCDropout,
            'knn':              KNN.KNNSVM,
            'bceaeknn':         KNN.BCEKNNSVM,
            'mseaeknn':         KNN.MSEKNNSVM,
            'vaeaeknn':         KNN.VAEKNNSVM,
            'binclass':         BinClass.BinaryClassifier,
            'sd_threshold':     SD.SDThreshold,
            'deep_ensemble':    DE.DeepEnsemble,
            'odin':             ODIN.ODIN,
            'reconst_thresh':   RE.ReconstructionThreshold,
            'pixelcnn':         PCNN.PixelCNN,
            'openmax':          OM.OpenMax,
        }

    def get_method(self, name, args):
        elements = name.split('/')
        try:
            instance = self.all_methods[elements[0]](args)
        except KeyError:
            print("CONFIG ERROR: We don't recognise the method name {}".format(elements[0]))
        if len(elements) > 1:
            instance.default_model = int(elements[1])
        return instance
    
    def generate(self,args):
        pass
    