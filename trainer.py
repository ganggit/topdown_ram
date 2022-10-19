import os
import time
import shutil
import pickle

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from model import RecurrentAttention
from utils import AverageMeter


class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.is_lstm = config.is_lstm
        self.is_context = config.is_context

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            # self.num_valid = len(self.valid_loader.sampler.indices)
            if hasattr(self.valid_loader.sampler, 'indices'):
                self.num_valid = len(self.valid_loader.sampler.indices)
            else:    # for cluttered image
                self.num_valid = len(self.valid_loader.sampler.data_source.labels)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 11 
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
            self.is_lstm
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.init_lr
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience
        )

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                )
            )

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        nrows = 2
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # add by Gang
                W = x.shape[2]
                H = x.shape[3]
                x_sub = torch.nn.functional.interpolate(x.detach().clone(), size=[2*self.patch_size,2*self.patch_size], mode='nearest', align_corners=None)
                scores = self.model.metacontrollor(x_sub)
                loc_predict = torch.max(scores, 1)[1]
                irow = torch.div(loc_predict, nrows)
                icol = torch.remainder(loc_predict, nrows)
                intervals = W/nrows
                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()
                for bindx in range(self.batch_size):
                    l_t[bindx, 0] = torch.FloatTensor(1, 1).uniform_(irow[bindx]*intervals, (1+irow[bindx])*intervals).to(self.device)
                    l_t[bindx, 1] = torch.FloatTensor(1, 1).uniform_(icol[bindx]*intervals, (1+icol[bindx])*intervals).to(self.device)
                l_t = (l_t.detach()/W)*2-1

                if self.is_lstm and self.is_context:    
                    h_t, l_t = self.reset()
                    state_h = self.model.cnet(x)
                    # state_h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                    state_c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                    h_t = (state_h.unsqueeze(0).to(self.device), state_c.to(self.device))
                else:
                    # initialize it with zeros 
                    state_h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                    state_c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                    h_t = (state_h.to(self.device), state_c.to(self.device))                  
                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                log_pi = []
                baselines = []
                log_pbs = [] 

                for t in range(self.num_glimpses):
                    # forward pass through model
                    if (t+1)%3 !=0:
                        h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                        # store
                        # locs.append(l_t[0:9])
                        baselines.append(b_t)
                        log_pi.append(p)
                    else: # doing the classification every 3 glimples
                        # last iteration
                        h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                        log_pi.append(p)
                        baselines.append(b_t)
                        log_pbs.append(log_probas)

                locs.append(l_t[0:9])
                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)

                logits = torch.stack(log_pbs).transpose(1, 0)
                # calculate reward
                predicted = torch.max(logits, 2)[1]
                R = (predicted.detach() == y).float()
                # R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                
                # get the classification mask
                mask = torch.ones(y.shape, dtype=torch.bool).to(self.device)
                idx = y==10   
                for midx in range(mask.shape[0]):
                    maskindex = idx[midx, :].nonzero()
                    if len(maskindex)>0:
                        mask[midx, maskindex[0]+1:] = False
                # 
                
                yidx = y.reshape(-1).unsqueeze(1)
                logp = torch.gather(logits.reshape(-1, self.num_classes), 1, yidx)
                # loss_action = F.nll_loss(logits.reshape(-1, self.num_classes), y.reshape(-1))
                loss_action = -(logp*(mask.clone().view(-1, 1))).mean()
                # yhat = torch.softmax(logits.reshape(-1, self.num_classes),  dim=-1)
                # loss_action = torch.nn.CrossEntropyLoss()(yhat*(mask.reshape(-1).unsqueeze(1).repeat(1, self.num_classes)), y.reshape(-1)*mask.reshape(-1))
                # loss_action = F.nll_loss(logits.reshape(-1, self.num_classes)*(mask.reshape(-1).unsqueeze(1).repeat(1, self.num_classes)), y.reshape(-1)*mask.reshape(-1))
                # get the global mask
                for midx in range(mask.shape[0]):
                    tmp =  1-R[midx, :]
                    maskindex = tmp.nonzero()
                    if len(maskindex)>0:
                        mask[midx, maskindex[0]+1:] = False 

                gmask =  torch.zeros([y.shape[0], self.num_glimpses], dtype=torch.bool, device=self.device) 
                for it in range(R.shape[1]):
                    gmask[:, 3*it: 3*(it+1)]  = mask[:, it].unsqueeze(1).repeat(1, 3)    

                Rewards = torch.zeros([y.shape[0], self.num_glimpses], dtype=torch.float32)
                for it in range(R.shape[1]):
                    Rewards[:, 3*(it+1) - 1]  = R[:, it] # * mask[:, it] #mask out
                for it in range(self.num_glimpses-1):
                    Rewards[:, it+1] = Rewards[:, it+1] + Rewards[:, it]

                # R = Rewards.to(self.device)     
                R = Rewards[:, -1].to(self.device).unsqueeze(1).repeat(1, self.num_glimpses)                     
                # compute losses for differentiable modules

                loss_baseline = F.mse_loss(baselines, Rewards.to(self.device).detach())
                # loss_baseline = F.mse_loss(baselines, R.detach())
                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi * adjusted_reward*(gmask.float()), dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)


                # add the meta controller
                values = torch.zeros(scores.shape).to(self.device)
                for bindx in range(self.batch_size):
                    values[bindx, irow[bindx]*2+ icol[bindx]]  = 1 if R[bindx,-1]> 0 else 0

                values = values - torch.mean(values, 0)
                meta_loss = F.mse_loss(scores, values.detach())

                # sum up into a hybrid loss
                loss = loss_action + loss_baseline + loss_reinforce * 0.1 + meta_loss*0.02

                # compute accuracy
                correct = (predicted == y)#.float()
                val_pred = torch.sum(correct*(y!=10), 1)
                val_true = torch.sum(y!=10, 1)
                correct = (val_pred==val_true).float()
                acc = 100 * (correct.sum() / len(y))

                # detach
                h_state = h_t[0].detach()
                c_state = h_t[1].detach()
                h_t = (h_state, c_state)
                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    pickle.dump(
                        imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb")
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            return losses.avg, accs.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        nrows = 2
        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # add by Gang
            W = x.shape[2]
            H = x.shape[3]
            x_sub = torch.nn.functional.interpolate(x.detach().clone(), size=[2*self.patch_size,2*self.patch_size], mode='nearest', align_corners=None)
            scores = self.model.metacontrollor(x_sub)
            loc_predict = torch.max(scores, 1)[1]
            irow = torch.div(loc_predict, nrows)
            icol = torch.remainder(loc_predict, nrows)
            intervals = W/nrows
            plot = False
            if (epoch % self.plot_freq == 0) and (i == 0):
                plot = True

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            for bindx in range(self.batch_size):
                l_t[bindx, 0] = torch.FloatTensor(1, 1).uniform_(irow[bindx]*intervals, (1+irow[bindx])*intervals).to(self.device)
                l_t[bindx, 1] = torch.FloatTensor(1, 1).uniform_(icol[bindx]*intervals, (1+icol[bindx])*intervals).to(self.device)
            l_t = (l_t.detach()/W)*2-1

            if self.is_lstm:
                h_t, l_t = self.reset()
                # state_h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                state_h = self.model.cnet(x)
                state_c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                h_t = (state_h.unsqueeze(0).to(self.device), state_c.to(self.device))
            else:
                # initialize it with zeros 
                state_h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                state_c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
                h_t = (state_h.to(self.device), state_c.to(self.device)) 
            
            # extract the glimpses
            log_pi = []
            baselines = []
            log_pbs = []
            for t in range(self.num_glimpses ):
                if (t+1)%3 !=0:
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                else:
                    # last iteration
                    h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                    log_pbs.append(log_probas)
                # store
                baselines.append(b_t)
                log_pi.append(p)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            logits = torch.stack(log_pbs).transpose(1, 0)
            # calculate reward
            predicted = torch.max(logits, 2)[1]
            # calculate reward
            #R = (predicted.detach() == y).float()
            # predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            mask = torch.ones(y.shape, dtype=torch.bool)
            idx = y==10   
            for midx in range(mask.shape[0]):
                maskindex = idx[midx, :].nonzero()
                if len(maskindex)>0:
                    mask[midx, maskindex[0]+1:] = False
                tmp =  1-R[midx, :]
                maskindex = tmp.nonzero()
                if len(maskindex)>0:
                    mask[midx, maskindex[0]+1:] = False    

            Rewards = torch.zeros([y.shape[0], self.num_glimpses], dtype=torch.float32)
            for it in range(R.shape[1]):
                Rewards[:, 3*(it+1)-1]  = R[:, it]  
            for it in range(self.num_glimpses-1):
                Rewards[:, it+1] = Rewards[:, it+1] + Rewards[:, it]

            gmask =  torch.zeros([y.shape[0], self.num_glimpses], dtype=torch.bool, device=self.device) 
            for it in range(R.shape[1]):
                gmask[:, 3*it: 3*(it+1)]  = mask[:, it].unsqueeze(1).repeat(1, 3)    


            # R = Rewards.to(self.device)   
            R = Rewards[:, -1].to(self.device).unsqueeze(1).repeat(1, self.num_glimpses)
            # compute losses for differentiable modules
            # loss_action = F.nll_loss(log_probas, y)
            loss_action = F.nll_loss(logits.reshape(-1, self.num_classes), y.reshape(-1))
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # compute accuracy
            correct = (predicted[:, 1:] == y[:, 1:]).float()
            val_pred = torch.sum(correct*(y[:, 1:]!=10), 1)
            val_true = torch.sum(y[:, 1:]!=10, 1)
            correct = (val_pred==val_true).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg

    @torch.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)

            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
