import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import models as Models
import global_vars as Global
from utils.args import args

import categories.classifier_setup as CLSetup
import categories.kway_logistic_setup as KLogisticSetup
import categories.deep_ensemble_setup as DeepEnsembleSetup
import categories.ae_setup as AESetup
import categories.pixelcnn_setup as PCNNSetup

import utils.distributed as distrib 
import torch
import wandb
from utils.distributedproxysampler import DistributedProxySampler

if args.exp != 'model_ref':
    print('The exp is NOT model_ref!')

def needs_processing(args, dataset_class, models, suffix):
    """
        This function checks whether this model is already trained and can be skipped.
    """
    for model in models:
        for suf in suffix:
            home_path = Models.get_ref_model_path(args, model.__class__.__name__, dataset_class.__name__, model_setup=True, suffix_str=suf)
            hbest_path = os.path.join(home_path, 'model.best.pth.done')
            if not os.path.isfile(hbest_path):
                return True
    return False

if __name__ == "__main__":

    distrib.init_distributed_mode(args)

    if (distrib.is_main_process() and not args.no_wandb):
        wandb.init(project=args.wandb_project, entity="aber-wgr")

    # Set the seed for reproducibility.
    seed = args.seed + distrib.get_rank()
    torch.manual_seed(seed)

    num_tasks = distrib.get_world_size()
    global_rank = distrib.get_rank()
    
    if (distrib.is_main_process()  and not args.no_wandb):
        wandb.config.update(args)

    task_list = [
        # The list of models,   The function that does the training,    Can I skip-test?,   suffix of the operation.
        # The procedures that can be skip-test are the ones that we can determine
        # whether we have done them before without instantiating the network architecture or dataset.
        # saves quite a lot of time when possible.
        (Global.dataset_reference_classifiers, CLSetup.train_classifier,            True, ['base']),
        #(Global.dataset_reference_classifiers, KLogisticSetup.train_classifier,     True, ['KLogistic']),
        #(Global.dataset_reference_classifiers, DeepEnsembleSetup.train_classifier,  True, ['DE.%d'%i for i in range(5)]),
        (Global.dataset_reference_autoencoders, AESetup.train_BCE_AE,               False, []),
        (Global.dataset_reference_autoencoders, AESetup.train_MSE_AE,               False, []),
        #(Global.dataset_reference_waes, AESetup.train_BCE_WAE,                      False, []),
        #(Global.dataset_reference_waes, AESetup.train_MSE_WAE,                      False, []),
        (Global.dataset_reference_vaes, AESetup.train_variational_autoencoder,      False, []),
        #(Global.dataset_reference_pcnns, PCNNSetup.train_pixelcnn,                  False, []),
    ]

    # Do a for loop to run the training tasks.
    for task_id, (ref_list, train_func, skippable, suffix) in enumerate(task_list):
        target_datasets = ref_list.keys()
        print('Processing %d datasets.'%len(target_datasets))
        for dataset in target_datasets:
            print('Processing dataset %s with %d networks for %d-%s.'%(dataset, len(ref_list[dataset]), task_id, train_func.__name__))
            if skippable and not needs_processing(args, Global.all_datasets[dataset], ref_list[dataset], suffix=suffix):
                print('Skipped')
                continue
            ds = Global.all_datasets[dataset](drop_class = args.drop_class)
            for model_builder in ref_list[dataset]:
                model_builder.add('split_size',(int)(args.batch_size / 4))
                model = model_builder()
                print('Training %s'%(model.__class__.__name__))
                train_func(args, model, ds)
