import os
import torch

from utils.args import args
import global_vars as Global

import json
from json.decoder import JSONDecodeError
#########################################################
d1_tasks, d2_tasks, d3_tasks, method_tasks = [], [], [], []

json_file = args.exp
json_exists = os.path.isfile(json_file)


if (not json_file.endswith('.json')) or (not json_exists):
    print("Using default evaluation")
    d1_tasks     = ['MNIST']
    d2_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
    d3_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
    method_tasks     = [
                        # 'empty/0',
                        'confidence_check/0',
                        'prob_threshold/0',
                        ]
else:
    with open(json_file,"r") as fp:
        try:
            jx = json.load(fp)
            args.exp = jx['name']
            print("Loaded experiment:" + args.exp)
            d1_tasks = jx['d1_tasks']
            d2_tasks = jx['d2_tasks']
            d3_tasks = jx['d3_tasks']
            method_tasks = jx['method_tasks']
        except KeyError:
            print("Bad JSON file, check key headers")
            quit()
        except JSONDecodeError:
            print("Bad JSON file, check structure")
            quit()

# Construct the dataset cache
ds_cache = {}
results = []

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

                print ("Performing %s on %s vs. %s-%s"%(method, d1,d2,d3))

                if torch.ByteTensor(
                        [has_done_before(method, d1, d2, d3) or not ds_cache[d3].is_compatible(ds1) or d2 == d3 for d3 in d3_tasks]
                    ).all():
                    continue

                ds3 = ds_cache[args.D3]

                if not ds3.is_compatible(ds1):
                    print ('%s is not compatible with %s, skipping.'
                            %(ds3.name,
                              ds1.name))
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
