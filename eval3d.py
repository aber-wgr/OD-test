import os
import torch

from utils.args import args
import global_vars as Global
import copy
from setup.categories.ae_setup import get_vae_config
from datasets import SubDataset

#########################################################
"""
    Master Evaluation.
"""
d1_tasks, d2_tasks, d3_tasks, method_tasks = [], [], [], []

if args.exp == 'master':
    d1_tasks     = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10']
    #d2_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
    #d3_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
    d2_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10']
    d3_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10']
    method_tasks = [
                    'pixelcnn/0',
                    'mcdropout/0',
                    'prob_threshold/0',     'prob_threshold/1',
                    'score_svm/0',          'score_svm/1',
                    'logistic_svm/0',       'logistic_svm/1',
                    'openmax/0',            'openmax/1',
                    'binclass/0',           'binclass/1',
                    'deep_ensemble/0',      'deep_ensemble/1',
                    'odin/0',               'odin/1',
                    'reconst_thresh/0',     'reconst_thresh/1',
                    'knn/1', 'knn/2', 'knn/4', 'knn/8',
                    'bceaeknn/1', 'vaeaeknn/1', 'mseaeknn/1',
                    'bceaeknn/2', 'vaeaeknn/2', 'mseaeknn/2',
                    'bceaeknn/4', 'vaeaeknn/4', 'mseaeknn/4',
                    'bceaeknn/8', 'vaeaeknn/8', 'mseaeknn/8',
                    ]
########################################################
"""
    Test evaluation
"""
if args.exp == 'test-eval':
    d1_tasks     = ['MNIST']
    d2_tasks     = ['UniformNoise', 'NormalNoise']
    d3_tasks     = ['UniformNoise', 'NormalNoise']
    method_tasks     = [
                        'prob_threshold/0',
                        ]
########################################################
"""
    Default Evaluation
"""
if len(d1_tasks) == 0:
    d1_tasks     = ['MNIST']
    #d2_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
    #d3_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
    d2_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10']
    d3_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10']
    
    method_tasks     = [
                        'prob_threshold/0',
                        ]

# Construct the dataset cache
ds_cache = {}

for m in [d1_tasks, d2_tasks, d3_tasks]:
    for d in m:
        if not d in ds_cache:
            ds_cache[d] = Global.all_datasets[d]()

results = []
# If results exists already, just continue where left off.
results_path = os.path.join(args.experiment_path, 'results.pth')
if os.path.exists(results_path) and not args.force_run:
    print ("Loading previous checkpoint")
    results = torch.load(results_path)

def has_done_before(method, d1, d2, d3):
    for m, ds, dm, dt, mid, a1, a2 in results:
        if m == method and ds == d1 and dm == d2 and dt == d3:
            return True
    return False

if __name__ == "__main__":
    for m in [d1_tasks, d2_tasks, d3_tasks]:
        for d in m:
            if d not in ds_cache:
                ds_cache[d] = Global.all_datasets[d](drop_class = args.drop_class)
    # If results exists already, just continue where left off.
    results_path = os.path.join(args.experiment_path, 'results.pth')
    if os.path.exists(results_path) and not args.force_run:
        print ("Loading previous checkpoint")
        results = torch.load(results_path)
    for d1 in d1_tasks:
        args.D1 = d1
        for method in method_tasks:
            BT = Global.get_method(method, args)
            for d2 in d2_tasks:
                args.D2 = d2

                print ("Performing %s on %s vs. %s"%(method, d1, d2))

                ds1 = ds_cache[args.D1]
                ds2 = ds_cache[args.D2]

                if not ds2.is_compatible(ds1):
                    print ('DS2:%s is not compatible with DS1:%s, skipping.'%(ds2.name, ds1.name))
                    continue


                if torch.ByteTensor([has_done_before(method, d1, d2, d3) or not ds_cache[d3].is_compatible(ds1) or d2 == d3 for d3 in d3_tasks]).all():
                    continue
                

                valid_mixture = None

                if not method.startswith('binclass'):
                    # Stage 1: Propose H
                    d1_train = ds1.get_D1_train()
                    BT.propose_H(d1_train)

                    # Stage 2: Train for h \in H
                    d1_valid = ds1.get_D1_valid()
                    valid_mixture = None
                    if(args.unseen_class_test):
                        # if we're running in "unseen class" mode, we validate on the dropped class in d1, not on d2
                        # we can use all the dropped from the training and validation sets, because they were not used in training and we know they're OOD
                        d1_valid_dropped = ds1.get_D1_valid_dropped()
                        d1_train_dropped = ds1.get_D1_train_dropped()

                        # Adjust the sizes.
                        d1_len = len(d1_valid)
                        d1_valid_len = len(d1_valid_dropped)
                        d1_train_len = len(d1_train_dropped)
                        final_len = min(d1_len, d1_valid_len + d1_train_len)
                        
                        d1_valid.trim_dataset(final_len)
                        # we can't use trim_dataset on a concatdataset
                        if(d1_valid_len + d1_train_len > final_len):
                            ratio = (d1_valid_len + d1_train_len)  / final_len
                            new_valid_len = d1_valid_len * ratio
                            new_train_len = d1_train_len * ratio
                            while (new_valid_len + new_train_len > final_len):
                                new_train_len = new_train_len - 1
                                
                            d1_valid_len = new_valid_len
                            d1_train_len = new_train_len
                        d1_valid_dropped.trim_dataset(d1_valid_len)
                        d1_train_dropped.trim_dataset(d1_train_len)

                        valid_mixture = d1_valid + d1_valid_dropped + d1_train_dropped
                    else:
                        d2_valid = ds2.get_D2_valid(ds1)

                        # Adjust the sizes.
                        d1_valid_len = len(d1_valid)
                        d2_valid_len = len(d2_valid)
                        final_len = min(d1_valid_len, d2_valid_len)
                        print("Adjusting %s and %s to %s"%(d1_valid_len,
                                                        d2_valid_len,
                                                        final_len))
                        d1_valid.trim_dataset(final_len)
                        d2_valid.trim_dataset(final_len)
                        valid_mixture = d1_valid + d2_valid
                        
                else:
                    print('Binary evaluation mode')
                    # There's no stage one; the method would do everything in the 
                    # second stage.

                    # Get the first split. Overwrite the label
                    d1_train = ds1.get_D1_train()
                    d1_train.label = 0
                    cls_name = d1_train.name

                    # Stage 2: Train for h \in H
                    d1_valid = ds1.get_D1_valid()
                    valid_mixture = None
                    if(args.unseen_class_test):
                        # if we're running in "unseen class" mode, we validate on the dropped class in d1, not on d2
                        # we can use all the dropped from the training and validation sets, because they were not used in training and we know they're OOD
                        d1_valid_dropped = ds1.get_D1_valid_dropped()
                        d1_train_dropped = ds1.get_D1_train_dropped()

                        # Adjust the sizes.
                        d1_valid_len = len(d1_valid_dropped)
                        d1_train_len = len(d1_train_dropped)
                        final_len = min(d1_valid_len, d1_valid_len + d1_train_len)
                        
                        d1_valid.trim_dataset(final_len)
                        # we can't use trim_dataset on a concatdataset
                        if(d1_valid_len + d1_train_len > final_len):
                            ratio = (d1_valid_len + d1_train_len)  / final_len
                            new_valid_len = d1_valid_len * ratio
                            new_train_len = d1_train_len * ratio
                            while (new_valid_len + new_train_len > final_len):
                                new_train_len = new_train_len - 1
                                
                            d1_valid_len = new_valid_len
                            d1_train_len = new_train_len
                        d1_valid_dropped.trim_dataset(d1_valid_len)
                        d1_train_dropped.trim_dataset(d1_train_len)

                        valid_mixture = d1_valid + d1_valid_dropped + d1_train_dropped
                    else:
                        d2_valid = ds2.get_D2_valid(ds1)

                        # Adjust the sizes.
                        d1_valid_len = len(d1_valid)
                        d2_valid_len = len(d2_valid)
                        final_len = min(d1_valid_len, d2_valid_len)
                        print("Adjusting %s and %s to %s"%(d1_valid_len,
                                                        d2_valid_len,
                                                        final_len))
                        d1_valid.trim_dataset(final_len)
                        d2_valid.trim_dataset(final_len)
                        valid_mixture = d1_valid + d2_valid

                 

                train_acc = BT.train_H(valid_mixture)

                for d3 in d3_tasks:
                    args.D3 = d3

                    if d2 == d3:
                        print ("Skipping, d2==d3")
                        continue

                    print ("Performing %s on %s vs. %s-%s"%(method,d1,d2,d3))

                    if has_done_before(method, d1, d2, d3):
                        print ("Skipped, has been done before.")
                        continue

                    ds3 = ds_cache[args.D3]

                    if not ds3.is_compatible(ds1):
                        print ('DS3:%s is not compatible with DS1:%s, skipping.'
                                %(ds3.name,
                                  ds1.name))
                        continue

                    # Stage 3: Eval h on test data of d3
                    d1_test = ds1.get_D1_test()
 
                    if(args.unseen_class_test):
                        # if we're running in "unseen class" mode, we validate on the dropped class in d1, not on d2
                        # we can use all the dropped from the training and validation sets, because they were not used in training and we know they're OOD
                        d2_test = ds1.get_D1_test_dropped()
                    else:
                        if(args.interpolate_shift):
                            # interpolate shift mode is intended to test the method against domain shift
                            # we start by training a variational autoencoder on d2 and d3 shuffled together
                            # then we generate test points from d2 and d3, put them into the VAE encoder, and interpolate between them to generate a new latent space point
                            # finally, we decode the latent space point and see if it's classified correctly
                            print("Interpolate shift mode")
                            print("Training VAE on %s and %s"%(d1, d3))
                            
                            # D2 and D3 can potentially be "fake" data, generated by noise functions
                            # if they are, we use the test version
                            
                            d2_train_basis = ds1.get_D1_train()
                                
                            if(hasattr(d3, 'get_D2_train')):
                                d3_train_basis = ds3.get_D2_train(ds1)
                            else:
                                d3_train_basis = ds3.get_D2_test(ds1)

                            # Combine the sets
                            d2_train_concat = d2_train_basis + d3_train_basis
                            d2_train_basis = SubDataset('%s-%s'%(args.D1, args.D3), '%s-%s'%(args.D1, args.D3), d2_train_concat, torch.arange(len(d2_train_concat)).int())

                            # Copy the VAE
                            autoencoder_model = Global.dataset_reference_vaes[ds1.base_name][0]
                            autoencoder_model = copy.deepcopy(autoencoder_model)
                            
                            # Set up the config
                            autoencoder_config = get_vae_config(args, autoencoder_model(), d2_train_basis)
                            autoencoder_config.max_epoch = 30
                            autoencoder_config.name = 'autoencoder_%s_%s'%(d2_train_basis.name, autoencoder_model.preferred_name())

                            autoencoder_trainer = IterativeTrainer(autoencoder_config, args)

                            autoencoder_home_path = Models.get_ref_model_path(args, "combined_" + autoencoder_config.model.__class__.__name__, d1 + "_" + d2, model_setup=True, suffix_str='base')
                            autoencoder_hbest_path = os.path.join(autoencoder_home_path, 'model.best.pth')

                            best_loss = 999999999

                            if not os.path.isfile(autoencoder_hbest_path+".done"):
                                best_accuracy = -1
                                for epoch in range(1, autoencoder_config.max_epoch+1):

                                    print("Epoch " + str(epoch))

                                    # Track the learning rates.
                                    lrs = [float(param_group['lr']) for param_group in autoencoder_config.optim.param_groups]
                                    autoencoder_config.logger.log('LRs', lrs, epoch)
                                    autoencoder_config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]
                                    
                                    torch.set_grad_enabled(True)
                                    # run the samples
                                    for i, (sample, label) in enumerate(autoencoder_config.train_loader):
                                        x = sample.to(args.device)
                                        
                                        torch.set_grad_enabled(True)

                                        # run the autoencoder
                                        autoencoder_config.model.train()
                                        autoencoder_config.optim.zero_grad()
                                        
                                        output = autoencoder_config.model(x)
                                    
                                        # calculate the base VAE loss (reconstruction loss + KL loss)
                                        loss = autoencoder_config.criterion(output, x)
                            
                                        loss.backward()
                                        autoencoder_config.optim.step()

                                        torch.set_grad_enabled(False)
                                        autoencoder_trainer.run_epoch(epoch, phase='test')
                                        
                                        test_loss = autoencoder_config.logger.get_measure('test_loss').mean_epoch()

                                    # Save the logger for future reference.
                                    torch.save(autoencoder_config.logger.measures, os.path.join(autoencoder_home_path, 'logger.pth'))

                                    if args.save and test_loss < best_loss:
                                        print('Updating the on file model with %s'%('%.4f'%test_loss))
                                        best_loss = test_loss
                                        torch.save(autoencoder_config.model.state_dict(), autoencoder_hbest_path)
                                                
                        
                            torch.save({'finished':True}, autoencoder_hbest_path + ".done")

                            print("Loading the best model.")
                            autoencoder_config.model.load_state_dict(torch.load(autoencoder_hbest_path))
                            autoencoder_config.model.eval()

                            # Now we have a trained VAE, we can generate test points from d2 and d3, and interpolate between them

                            # Get the test sets
                            d2_test_base = ds2.get_D2_test(ds1)
                            d3_test_base = ds3.get_D2_test(ds1)
                            
                            # Adjust the sizes.
                            d2_test_len = len(d2_test_base)
                            d3_test_len = len(d3_test_base)

                            # adjust the sizes to be the same

                            final_len = min(d2_test_len, d3_test_len)

                            print("Adjusting %s and %s to %s"%(d2_test_len,
                                                            d3_test_len,
                                                            final_len))
                            
                            d2_test_base.trim_dataset(final_len)
                            d3_test_base.trim_dataset(final_len)

                            # Now run through the test sets, and generate a new point for each pair of points
                            # First, we need to get the latent space representations of the test points
                            # We'll store the latent space representations in a list

                            d2_test_latent = []

                            # run the samples

                            for i, (sample, label) in enumerate(d2_test_base):
                                x = sample.to(args.device)

                                # run the autoencoder
                                autoencoder_config.model.eval()

                                # get the latent space representation
                                latent = autoencoder_config.model.encode(x)

                                # add it to the list
                                d2_test_latent.append(latent)

                            # repeat for d3

                            d3_test_latent = []

                            # run the samples

                            for i, (sample, label) in enumerate(d3_test_base):
                                x = sample.to(args.device)

                                # run the autoencoder
                                autoencoder_config.model.eval()

                                # get the latent space representation
                                latent = autoencoder_config.model.encode(x)

                                # add it to the list
                                d3_test_latent.append(latent)

                            # Now we get the interpolated points
                            # Because we're looking at domain shift, we will run from 0 to 1 over the length of the test set

                            for i in range(final_len):
                                # get the latent space representations of the two points
                                latent1 = d2_test_latent[i]
                                latent2 = d3_test_latent[i]

                                # interpolate between them
                                interpolated = latent1 + (latent2 - latent1) * i / final_len

                                # decode the interpolated point
                                decoded = autoencoder_config.model.decode(interpolated)

                                # add the decoded point to the test set
                                d2_test.append(decoded, 0)
                        else:
                            d2_test = ds3.get_D2_test(ds1)
                    

                    # Adjust the sizes.
                    d1_test_len = len(d1_test)
                    d2_test_len = len(d2_test)
                    final_len = min(d1_test_len, d2_test_len)
                    print("Adjusting %s and %s to %s"%(d1_test_len,
                                                    d2_test_len,
                                                    final_len))
                    d1_test.trim_dataset(final_len)
                    d2_test.trim_dataset(final_len)
                    test_mixture = d1_test + d2_test
                    print("Final test size: %d+%d=%d"%(len(d1_test), len(d2_test), len(test_mixture)))

                    test_acc = BT.test_H(test_mixture)
                    results.append((method, d1, d2, d3, BT.method_identifier(), train_acc, test_acc))

                    # Take a snapshot after each experiment.
                    torch.save(results, results_path)

    for i, (m, ds, dm, dt, mi, a_train, a_test) in enumerate(results):
        print ('%d\t%s\t%15s\t%-15s\t%.2f%% / %.2f%%'%(i, m, '%s-%s'%(ds, dm), dt, a_train*100, a_test*100))    
