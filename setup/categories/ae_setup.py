import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,WeightedRandomSampler,RandomSampler

import models as Models
import global_vars as Global
from utils.iterative_trainer import IterativeTrainer, IterativeTrainerConfig
from utils.logger import Logger
from datasets import MirroredDataset

from models.autoencoders import VAE_Loss

import utils.distributed as distrib 
#import wandb
from utils.distributedproxysampler import DistributedProxySampler

def get_ae_config(args, model, domain, BCE_Loss):
    print("Preparing training D1 for %s"%(domain.name))

    dataset = domain.get_D1_train()

    torch.manual_seed(5555)

    # 80%, 20% for local train+test
    train_ds, valid_ds = dataset.split_dataset(0.8)

    seed = args.seed + distrib.get_rank()
    torch.manual_seed(seed)

    if dataset.name in Global.datasetStore.mirror_augment:
        print("Mirror augmenting %s"%dataset.name)
        new_train_ds = train_ds + MirroredDataset(train_ds)
        train_ds = new_train_ds

    train_sampler = RandomSampler(train_ds, replacement=False)
    val_sampler = RandomSampler(valid_ds,replacement=False)
    all_sampler = RandomSampler(dataset,replacement=False)

    if distrib.is_dist_avail_and_initialized():
        train_sampler = DistributedProxySampler( train_sampler,
            num_replicas=distrib.get_world_size(),
            rank=distrib.get_rank(), drop_last=True
        )
        val_sampler = DistributedProxySampler( val_sampler,
            num_replicas=distrib.get_world_size(),
            rank=distrib.get_rank(), drop_last=True
        )
        all_sampler = DistributedProxySampler( all_sampler,
            num_replicas=distrib.get_world_size(),
            rank=distrib.get_rank(), drop_last=True
        )

    # Initialize the multi-threaded loaders.
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    all_loader   = DataLoader(dataset,  sampler=all_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    
    # Set up the model
    #model = model.to(args.device)

    model_without_ddp = model

    if distrib.is_dist_avail_and_initialized():
        # if we're in distibuted mode, there are a number of possible configurations.
        if len(args.gpulist) > 1:
            # we have more than one GPU per node, which means we're probably running the multi-GPU versions of the classifiers.
            print("GPUs per process > 1, setting up DDP")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
        else:
            # we have one GPU per node, which means we're probably running the single-GPU versions of the classifiers.
            print("GPUs per process == 1, setting up DDP for GPU" + str(args.gpulist[0]))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpulist[0]])
    else:
        if len(args.gpulist) == 1:
            # we are not in distributed mode and we have one GPU, which means we're probably running the single-GPU versions of the classifiers.
            print("Single GPU mode")
            model = model.to(args.gpulist[0])

    # Set up the criterion
    criterion = None
    if BCE_Loss:
        criterion = nn.BCEWithLogitsLoss().to(args.device)
    else:
        criterion = nn.MSELoss().to(args.device)
        model.default_sigmoid = True

    # Set up the config
    config = IterativeTrainerConfig()

    config.name = 'autoencoder_%s_%s'%(dataset.name, model.preferred_name())

    config.train_loader = train_loader
    config.valid_loader = valid_loader
    config.phases = {
                    'train':   {'dataset' : train_loader,  'backward': True},
                    'test':    {'dataset' : valid_loader,  'backward': False},
                    'all':     {'dataset' : all_loader,    'backward': False},                        
                    }
    config.criterion = criterion
    config.classification = False
    config.cast_float_label = False
    config.autoencoder_target = True
    config.stochastic_gradient = True
    config.sigmoid_viz = BCE_Loss
    config.model = model
    config.model_without_ddp = model_without_ddp
    config.logger = Logger()

    config.optim = optim.Adam(model.parameters(), lr=1e-3)
    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
    config.max_epoch = 120
    
    if hasattr(model_without_ddp, 'train_config'):
        model_train_config = model_without_ddp.train_config()
        for key, value in model_train_config.items():
            print('Overriding config.%s'%key)
            config.__setattr__(key, value)

    return config

def get_vae_config(args, model, domain):
    print("Preparing training D1 for %s"%(domain.name))

    torch.manual_seed(5555)

    dataset = domain.get_D1_train()

    # 80%, 20% for local train+test
    train_ds, valid_ds = dataset.split_dataset(0.8)

    seed = args.seed + distrib.get_rank()
    torch.manual_seed(seed)

    if dataset.name in Global.datasetStore.mirror_augment:
        print("Mirror augmenting %s"%dataset.name)
        new_train_ds = train_ds + MirroredDataset(train_ds)
        train_ds = new_train_ds

    train_sampler = RandomSampler(train_ds, replacement=False)
    val_sampler = RandomSampler(valid_ds,replacement=False)
    all_sampler = RandomSampler(dataset,replacement=False)

    if distrib.is_dist_avail_and_initialized():
        train_sampler = DistributedProxySampler( train_sampler,
            num_replicas=distrib.get_world_size(),
            rank=distrib.get_rank(), drop_last=True
        )
        val_sampler = DistributedProxySampler( val_sampler,
            num_replicas=distrib.get_world_size(),
            rank=distrib.get_rank(), drop_last=True
        )
        all_sampler = DistributedProxySampler( all_sampler,
            num_replicas=distrib.get_world_size(),
            rank=distrib.get_rank(), drop_last=True
        )

    # Initialize the multi-threaded loaders.
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    all_loader   = DataLoader(dataset,  sampler=all_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    # Set up the model
    #model = model.to(args.device)

    # Set up the criterion
    criterion = VAE_Loss(model)

    model_without_ddp = model

    if distrib.is_dist_avail_and_initialized():
        # if we're in distibuted mode, there are a number of possible configurations.
        if len(args.gpulist) > 1:
            # we have more than one GPU per node, which means we're probably running the multi-GPU versions of the classifiers.
            print("GPUs per process > 1, setting up DDP")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
        else:
            # we have one GPU per node, which means we're probably running the single-GPU versions of the classifiers.
            print("GPUs per process == 1, setting up DDP for GPU" + str(args.gpulist[0]))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpulist[0]])
    else:
        if len(args.gpulist) == 1:
            # we are not in distributed mode and we have one GPU, which means we're probably running the single-GPU versions of the classifiers.
            print("Single GPU mode")
            model = model.to(args.gpulist[0])

    # Set up the config
    config = IterativeTrainerConfig()

    config.name = 'vae_%s_%s'%(dataset.name, model.preferred_name())

    config.train_loader = train_loader
    config.valid_loader = valid_loader
    config.phases = {
                    'train':   {'dataset' : train_loader,  'backward': True},
                    'test':    {'dataset' : valid_loader,  'backward': False},
                    'all':     {'dataset' : all_loader,    'backward': False},                        
                    }
    config.criterion = criterion
    config.classification = False
    config.cast_float_label = False
    config.autoencoder_target = True
    config.stochastic_gradient = True
    config.sigmoid_viz = False
    config.model = model
    config.model_without_ddp = model_without_ddp
    config.logger = Logger()

    config.optim = optim.Adam(model.parameters(), lr=1e-3)
    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
    config.max_epoch = 120
    
    if hasattr(model_without_ddp, 'train_config'):
        model_train_config = model_without_ddp.train_config()
        for key, value in model_train_config.items():
            print('Overriding config.%s'%key)
            config.__setattr__(key, value)

    return config

def train_BCE_AE(args, model, domain):
    train_autoencoder(args, model, domain, BCE_Loss=True)

def train_MSE_AE(args, model, domain):
    train_autoencoder(args, model, domain, BCE_Loss=False)

def train_autoencoder(args, model, domain, BCE_Loss):
    if BCE_Loss:
        model.netid = "BCE." + model.netid
    else:
        model.netid = "MSE." + model.netid

    config = get_ae_config(args, model, domain, BCE_Loss=BCE_Loss)

    home_path = Models.get_ref_model_path(args, model.__class__.__name__, domain.name, model_setup=True, suffix_str=model.netid)
    hbest_path = os.path.join(home_path, 'model.best.pth')
    hlast_path = os.path.join(home_path, 'model.last.pth')

    if not os.path.isdir(home_path):
        os.makedirs(home_path)

    if not os.path.isfile(hbest_path+".done"):
        trainer = IterativeTrainer(config, args)
        print('Training from scratch')
        best_loss = 999999999
        for epoch in range(1, config.max_epoch+1):

            # Track the learning rates.
            lrs = [float(param_group['lr']) for param_group in config.optim.param_groups]
            config.logger.log('LRs', lrs, epoch)
            config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]
            
            # One epoch of train and test.
            trainer.run_epoch(epoch, phase='train')
            trainer.run_epoch(epoch, phase='test')

            if(distrib.is_main_process()):

                train_loss = config.logger.get_measure('train_loss').mean_epoch()
                test_loss = config.logger.get_measure('test_loss').mean_epoch()

                config.scheduler.step(train_loss)

            
                # Save the logger for future reference.
                torch.save(config.logger.measures, os.path.join(home_path, 'logger.pth'))

                # Saving a checkpoint. Enable if needed!
                # if args.save and epoch % 10 == 0:
                #     print('Saving a %s at iter %s'%('snapshot', '%d'%epoch))
                #     torch.save(config.model.state_dict(), os.path.join(home_path, 'model.%d.pth'%epoch))

                if args.save and test_loss < best_loss:
                    print('Updating the on file model with %s'%('%.4f'%test_loss))
                    best_loss = test_loss
                    distrib.save_on_master(config.model.state_dict(), hbest_path)
                    # why barrier here? Because otherwise, if the last epoch is best acc,
                    # GPU0 saves and the others drop through to loading before it's finished.
                    torch.distributed.barrier() 
        
        distrib.save_on_master({'finished':True}, hbest_path + ".done")
        distrib.save_on_master(config.model.state_dict(), hlast_path)

    else:
        print("Skipping %s"%(home_path))

def train_variational_autoencoder(args, model, domain):
    
    config = get_vae_config(args, model, domain)
    
    home_path = Models.get_ref_model_path(args, config.model_without_ddp.__class__.__name__, domain.name, model_setup=True, suffix_str=model.netid)
    hbest_path = os.path.join(home_path, 'model.best.pth')
    hlast_path = os.path.join(home_path, 'model.last.pth')

    if not os.path.isdir(home_path):
        os.makedirs(home_path,exist_ok=True)

    if not os.path.isfile(hbest_path+".done"):
        trainer = IterativeTrainer(config, args)
        print('Training from scratch')
        best_loss = 999999999
        for epoch in range(1, config.max_epoch+1):

            # Track the learning rates.
            lrs = [float(param_group['lr']) for param_group in config.optim.param_groups]
            config.logger.log('LRs', lrs, epoch)
            config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]
            
            # One epoch of train and test.
            trainer.run_epoch(epoch, phase='train')
            trainer.run_epoch(epoch, phase='test')
            if(distrib.is_main_process()):

                train_loss = config.logger.get_measure('train_loss').mean_epoch()
                test_loss = config.logger.get_measure('test_loss').mean_epoch()

                config.scheduler.step(train_loss)

                
                # Save the logger for future reference.
                torch.save(config.logger.measures, os.path.join(home_path, 'logger.pth'))

                # Saving a checkpoint. Enable if needed!
                # if args.save and epoch % 10 == 0:
                #     print('Saving a %s at iter %s'%('snapshot','%d'%epoch))
                #     torch.save(config.model.state_dict(), os.path.join(home_path, 'model.%d.pth'%epoch))

                if args.save and test_loss < best_loss:
                    print('Updating the on file model with %s'%('%.4f'%test_loss))
                    best_loss = test_loss
                    distrib.save_on_master(config.model.state_dict(), hbest_path)
            # why barrier here? Because otherwise, if the last epoch is best acc,
            # GPU0 saves and the others drop through to loading before it's finished.
            torch.distributed.barrier() 
        
        distrib.save_on_master({'finished':True}, hbest_path+".done")
        distrib.save_on_master(config.model.state_dict(), hlast_path)

    else:
        print("Skipping %s"%(home_path))
