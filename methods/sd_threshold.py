import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger
import os
from os import path

from methods import AbstractMethodInterface, AbstractModelWrapper, SVMLoss
from datasets import MirroredDataset
import global_vars as Global

class SDModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
        For Base Threshold, we simply apply f(x) = sign(t-x), where x is the max probability and t is the threshold.
        We learn the the threshold with an SVM loss and a zero margin. This yields exactly the optimal threshold learning problem.
    """
    def __init__(self, base_model):
        super(SDModelWrapper, self).__init__(base_model)
        self.base_model = base_model
        self.H = nn.Module()
        self.H.register_parameter('threshold', nn.Parameter(torch.Tensor([0.05]))) # initialize to sd=0.05 for faster convergence.

    def subnetwork_eval(self, x):
        base_output = self.base_model(x, softmax=False)
        # Get the std deviation of the logits (testing response variability)
        output = base_output.std(axis=1).unsqueeze_(1)
        return output

    def wrapper_eval(self, x):
        # Threshold hold the avg std deviation
        output = self.H.threshold - x
        return output
    
    def classify(self, x):
        #breakpoint()
        return (x > 0).long()

    def get_output_device(self):
        return self.base_model.get_output_device()

    def get_info(self,args):
        return self.base_model.get_info(args)


class SDThreshold(AbstractMethodInterface):
    def __init__(self, args):
        super(SDThreshold, self).__init__()
        self.base_model = None
        self.H_class = None
        self.args = args
 
        self.default_model = 0
        self.add_identifier = ""

    def method_identifier(self):
        output = "SD_Thresh"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def get_base_config(self, dataset):
        print("Preparing training D1 for %s"%(dataset.name))

        all_loader   = DataLoader(dataset,  batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=True)

        # Set up the criterion
        criterion = nn.NLLLoss().to(self.args.device)
        criterion.size_average = True

        # Set up the model
        import global_vars as Global
        model = Global.modelStore.get_ref_classifier(dataset.name)[self.default_model]().to(self.args.device)
        self.add_identifier = model.__class__.__name__
        if hasattr(model, 'preferred_name'):
            self.add_identifier = model.preferred_name()

        # Set up the config
        config = IterativeTrainerConfig()

        config.name = '%s-CLS'%(self.args.D1)
        config.phases = {
                        'all':     {'dataset' : all_loader,    'backward': False},                        
                        }
        config.criterion = criterion
        config.classification = True
        config.cast_float_label = False
        config.stochastic_gradient = True
        config.model = model
        config.optim = None
        config.autoencoder_target = False
        config.logger = Logger()
        return config

    def propose_H(self, dataset):
        config = self.get_base_config(dataset)

        from models import get_ref_model_path
        h_path = get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name)
        best_h_path = path.join(h_path, 'model.best.pth')

        trainer = IterativeTrainer(config, self.args)

        if not path.isfile(best_h_path):      
            raise NotImplementedError("Please use model_setup to pretrain the networks first!")
        else:
            print('Loading H1 model from %s'%best_h_path)
            config.model.load_state_dict(torch.load(best_h_path))
        
        trainer.run_epoch(0, phase='all')
        test_average_acc = config.logger.get_measure('all_accuracy').mean_epoch(epoch=0)
        print("All average accuracy %s"%(test_average_acc*100))

        self.base_model = config.model
        self.base_model.eval()

    def get_H_config(self, dataset, will_train=True):
        print("Preparing training D1+D2 (H)")
        print("Mixture size: %s"%(len(dataset)))

        # 80%, 20% for local train+test
        train_ds, valid_ds = dataset.split_dataset(0.8)

        if self.args.D1 in Global.datasetStore.mirror_augment:
            print("Mirror augmenting %s"%self.args.D1)
            new_train_ds = train_ds + MirroredDataset(train_ds)
            train_ds = new_train_ds

        # Initialize the multi-threaded loaders.
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)

        # Set up the criterion
        # To make the threshold learning, actually threshold learning
        # the margin must be set to 0.
        criterion = SVMLoss(margin=0.0).to(self.args.device)

        # Set up the model
        model = SDModelWrapper(self.base_model).to(self.args.device)

        old_valid_loader = valid_loader
        if will_train:
            # cache the subnetwork for faster optimization.
            from methods import get_cached
            from torch.utils.data.dataset import TensorDataset

            trainX, trainY = get_cached(model, train_loader, self.args.device)
            validX, validY = get_cached(model, valid_loader, self.args.device)

            new_train_ds = TensorDataset(trainX, trainY)
            x_center = trainX[trainY==0].mean()
            y_center = trainX[trainY==1].mean()
            init_value = (x_center+y_center)/2
            model.H.threshold.fill_(init_value)
            print("Initializing threshold to %.2f"%(init_value.item()))

            new_valid_ds = TensorDataset(validX, validY)

            # Initialize the new multi-threaded loaders.
            train_loader = DataLoader(new_train_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)
            valid_loader = DataLoader(new_valid_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)

            # Set model to direct evaluation (for cached data)
            model.set_eval_direct(True)

        # Set up the config
        config = IterativeTrainerConfig()

        base_model_name = self.base_model.__class__.__name__
        if hasattr(self.base_model, 'preferred_name'):
            base_model_name = self.base_model.preferred_name()

        config.name = '_%s[%s](%s-%s)'%(self.__class__.__name__, base_model_name, self.args.D1, self.args.D2)
        config.train_loader = train_loader
        config.valid_loader = valid_loader
        config.phases = {
                        'train':   {'dataset' : train_loader,  'backward': True},
                        'test':    {'dataset' : valid_loader,  'backward': False},
                        'testU':   {'dataset' : old_valid_loader, 'backward': False},                        
                        }
        config.criterion = criterion
        config.classification = True
        config.cast_float_label = True
        config.stochastic_gradient = True
        config.model = model
        config.optim = optim.Adagrad(model.H.parameters(), lr=1e-1, weight_decay=0)
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-1, min_lr=1e-8, factor=0.1, verbose=True)
        config.logger = Logger()
        config.max_epoch = 100

        return config

    def train_H(self, dataset):
        # Wrap the (mixture)dataset in SubDataset so to easily
        # split it later.
        from datasets import SubDataset
        dataset = SubDataset('%s-%s'%(self.args.D1, self.args.D2), dataset, torch.arange(len(dataset)).int())
        
        h_path = path.join(self.args.experiment_path, '%s'%(self.__class__.__name__),
                                                      '%d'%(self.default_model),
                                                      '%s_%s.pth'%(self.args.D1, self.args.D2))
        h_parent = path.dirname(h_path)
        if not path.isdir(h_parent):
            os.makedirs(h_parent)
        
        done_path = h_path + '.done'
        will_train = self.args.force_train_h or not path.isfile(done_path)

        h_config = self.get_H_config(dataset, will_train)

        trainer = IterativeTrainer(h_config, self.args)

        if will_train:
            print('Training from scratch')
            best_accuracy = -1
            #pdb.set_trace()
            trainer.run_epoch(0, phase='test')
            for epoch in range(1, h_config.max_epoch+1):
                trainer.run_epoch(epoch, phase='train')
                trainer.run_epoch(epoch, phase='test')

                train_loss = h_config.logger.get_measure('train_loss').mean_epoch()
                h_config.scheduler.step(train_loss)

                # Track the learning rates and threshold.
                lrs = [float(param_group['lr']) for param_group in h_config.optim.param_groups]
                h_config.logger.log('LRs', lrs, epoch)
                h_config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]

                test_average_acc = h_config.logger.get_measure('test_accuracy').mean_epoch()

                # Save the logger for future reference.
                #torch.save(h_config.logger.measures, path.join(h_parent, 'logger.%s-%s.pth'%(self.args.D1, self.args.D2)))

                if best_accuracy < test_average_acc:
                    print('Updating the on file model with %s'%(test_average_acc))
                    best_accuracy = test_average_acc
                    torch.save(h_config.model.H.state_dict(), h_path)
            
            torch.save({'finished':True}, done_path)

        # Load the best model.
        print('Loading H model from %s'%h_path)
        h_config.model.H.load_state_dict(torch.load(h_path))
        h_config.model.set_eval_direct(False)        
        
        trainer.run_epoch(0, phase='testU')
        test_average_acc = h_config.logger.get_measure('testU_accuracy').mean_epoch(epoch=0)
        print("Valid/Test average accuracy %s"%(test_average_acc*100))
        self.H_class = h_config.model
        self.H_class.eval()
        self.H_class.set_eval_direct(False)        
        return test_average_acc

    def test_H(self, dataset):
        dataset = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)
        correct = 0.0
        total_count = 0
        self.H_class.eval()
        for i, (image, label) in enumerate(dataset):

            # Get and prepare data.
            input, target = image.to(self.args.device), label.to(self.args.device)
            #print("target x:{}".format(target))
            prediction = self.H_class(input)
            classification = self.H_class.classify(prediction)

            correct += (classification.detach().view(-1) == target.detach().view(-1).long()).float().view(-1).sum()
            total_count += len(input)

            message = 'Accuracy %.4f'%(correct/total_count)       
        
        test_average_acc = correct/total_count
        print("Final Test average accuracy %s"%(test_average_acc*100))
        return test_average_acc.item()
