import errno

import timeit

import torch
import torch.nn.functional as F
import models as Models

import matplotlib.pyplot as plt

import os

"""
From https://en.wikipedia.org/wiki/Coefficient_of_determination
"""
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

class IterativeTrainerConfig(object):
    pass

class IterativeTrainer(object):
    def __init__(self, config, args):
        self.config = config
        self.args   = args
        self.device = args.device
        # Set the default behaviours if not set.
        defaults = {
            'classification': False,
            'cast_float_label': False,
            'autoencoder_target': False,
            'autoencoder_class': False,
            'stochastic_gradient': True,
            'sigmoid_viz': True,
        }
        for key, value in defaults.items():
            if not hasattr(self.config, key):
                print('Setting default value %s to %s'%(key, value))
                setattr(self.config, key, value)
    
    def run_epoch(self, epoch, phase='train'):
        # Retrieve the appropriate config.
        config      = self.config.phases[phase]
        dataset     = config['dataset']
        backward    = config['backward']
        phase_name  = phase
        print("Doing %s"%phase)

        model       = self.config.model
        criterion   = self.config.criterion
        optimizer   = self.config.optim
        logger      = self.config.logger
        stochastic  = self.config.stochastic_gradient
        classification = self.config.classification

        #print("self.config.name:" + self.config.name)
        home_path = Models.get_ref_model_path(self.args, model.__class__.__name__, self.config.name, model_setup=True, suffix_str="CCC")
        dump_path = os.path.join(home_path, 'dump')
        if not os.path.isdir(dump_path):
            os.makedirs(dump_path)

        # See the network to the target mode.
        if backward:
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        start_time = timeit.default_timer()

        # For full gradient optimization we need to rescale the loss
        # to calculate the gradient correctly.
        loss_scaler = 1
        if not stochastic:
            loss_scaler = 1./len(dataset.dataset)

        if backward and not stochastic:
            optimizer.zero_grad()

        for i, (image, label) in enumerate(dataset):
            if backward and stochastic:
                optimizer.zero_grad()

            # Get and prepare data.
            input, target, data_indices = image, None, None
            if torch.typename(label) == 'list':
                assert len(label) == 2, 'There should be two entries in the label'
                # Need to unpack the label. This is for when the data provider
                # has the cached flag enabled, therefore the y is now (y, idx).
                target, data_indices = label
            else:
                target = label
                    
            if self.config.autoencoder_target:
                target = input.clone()
                    
            if self.config.cast_float_label:
                target = target.float().unsqueeze(1)

            input, target = input.to(self.device), target.to(model.get_output_device())

            # Do a forward propagation and get the loss.
            prediction = None
            if data_indices is None:
                prediction = model(input)
            else:
                # Run in the cached mode. This is necessary to speed up
                # some of the underlying optimization procedures. It is not
                # always used though.
                prediction = model(input, indices=data_indices, group=phase_name)
            loss = criterion(prediction, target)

            if(self.args.dump_images):
                # pick one from the batch and output it
                
                #filename = phase_name + str(i) +"_epoch" + str(epoch) + ".png"
                #dump_file = os.path.join(dump_path,filename)
                #self.dump_image(input[0].cpu(),dump_file,True)

                if self.config.autoencoder_target:

                    home_path = Models.get_ref_model_path(self.args, model.__class__.__name__, self.config.name, model_setup=True, suffix_str="CCC")
                    dump_path = os.path.join(home_path, 'dump')
                    if not os.path.isdir(dump_path):
                        os.makedirs(dump_path)

                    filename = phase_name + str(i) +"_epoch" + str(epoch) + ".png"
                    dump_file = os.path.join(dump_path,filename)
                    self.dump_image(input[0].cpu(),dump_file,True)

                    # if this is an autoencoder run, also output the recreation for comparison
                    filename = phase_name + str(i) + "_epoch" + str(epoch) + "_target.png"
                    dump_file = os.path.join(dump_path,filename)
                    self.dump_image(prediction[0].cpu(),dump_file,True)


            if backward:
                if stochastic:
                    loss.backward()
                    optimizer.step()
                else:
                    nscaler = loss_scaler
                    if criterion.size_average:
                        nscaler = nscaler * len(input)
                    loss2 = loss * nscaler
                    loss2.backward()

            criterion.size_average = True
            # Compute various measure. Can be safely skipped.
            if not backward or not stochastic:
                if criterion.size_average:
                    loss.data.mul_(len(input))
            logger.log('%s_loss'%phase_name, loss.item(), epoch, i)
            message = '%s Batch loss %.3f'%(phase_name, loss.item())

            if classification:
                pred = []
                if prediction.size(1) == 1:
                    # For binary classification, we do this to make the
                    # prediction code consistent with the max.
                    pred = model.classify(prediction)
                else:
                    pred = prediction.max(1)[1]
                if stochastic and backward:
                    acc = (pred == target.long()).float().view(-1).mean().item()
                    logger.log('%s_accuracy'%phase_name, acc, epoch, i)
                else:
                    acc = (pred == target.long()).float().view(-1).sum().item()
                    logger.log('%s_accuracy'%phase_name, acc, epoch, i)
                    acc = acc/target.numel()
                message = '%s Accuracy %.2f'%(message, acc)
            else:
                # For regression tasks, use r2 loss
                acc = r2_loss(prediction,target).cpu()
                logger.log('%s_accuracy'%phase_name, acc, epoch, i)

            if backward and not stochastic:
                optimizer.step()

        if not backward or not stochastic:
            logger.get_measure('%s_loss'%phase_name).measure_normalizer = len(dataset.dataset)
            if classification:
                logger.get_measure('%s_accuracy'%phase_name).measure_normalizer = len(dataset.dataset)

        elapsed = timeit.default_timer() - start_time
        print('  %s Epoch %d in %.2fs' %(phase_name, epoch, elapsed))

    def dump_image(self,imageTensor,path,force_grayscale=False):
        image_n = imageTensor.permute(1,2,0)
        plt.figure(figsize=(4,4))
        plt.imshow(image_n.detach())
        if force_grayscale:
            plt.gray()
        plt.savefig(path)
        plt.close()
