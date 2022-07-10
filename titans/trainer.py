import torch
import wandb
import tqdm
import os

from ..titans.meters import ScoreMeter, LossMeter
from ..titans.earlystopping import EarlyStopping
from ..utils import set_seed, get_loaders
from ..utils import beaty_utils
from ..utils import cfg_utils


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(self.cfg.seed)
        self.resume = False
        self.cache = {'score': None, 'loss': None}

        self.device = torch.device(self.cfg.device)
        self.train_dl, self.val_dl = get_loaders(self.cfg)
        self.model, self.optimizer, self.scheduler = cfg_utils.get_setup(self.cfg,
                                                                         self.cfg.epochs,
                                                                         len(self.train_dl))

        self.train_score_meter = ScoreMeter(self.cfg)
        self.val_score_meter = ScoreMeter(self.cfg)

        self.train_loss_meter = LossMeter(self.cfg)
        self.val_loss_meter = LossMeter(self.cfg)

        self.early_stopping = EarlyStopping(self.train_loss_meter,
                                            self.val_loss_meter,
                                            max_step=self.cfg.stop_earlystopping_step)

        self.start_epoch = self.cfg.start_epoch
        if self.cfg.end_epoch == -1:
            self.end_epoch = self.cfg.epochs
        else:
            self.end_epoch = self.cfg.end_epoch

    def load_state_dict(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.resume = True
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def remember_state_dict(self, checkpoint_path, epoch, mode):
        self.cache[mode] = {'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict()
                            }, checkpoint_path

    def save_state_dict(self, mode):
        torch.save(*self.cache[mode])

    def main_loop(self, use_wandb=False, verb=True):
        if use_wandb:
            wandb.init(project=self.cfg.wandb_project, config=self.cfg, name=self.cfg.save_name)
            wandb.watch(self.model, log_freq=100)
            print('[*] wandb is watching')

        epoch_zeros_number = len(str(self.cfg.epochs))
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            # for beauty saving
            str_epoch = str(epoch).rjust(epoch_zeros_number, '0')

            # <<<<< TRAIN >>>>>
            self.train_epoch()
            train_loss = self.train_loss_meter.get_mean_loss()
            train_scores = self.train_score_meter.get_mean_stats()

            # <<<<< EVAL >>>>>
            self.eval_epoch()
            val_loss = self.val_loss_meter.get_mean_loss()
            val_scores = self.val_score_meter.get_mean_stats()

            # <<<<< LOGGING >>>>>
            metrics = {'train_loss': train_loss,
                       'val_loss': val_loss,
                       'lr': self.scheduler.get_last_lr()[-1]}
            for key in list(train_scores.keys()):
                metrics[f'train_{key}'] = train_scores[key]
            for key in list(val_scores.keys()):
                metrics[f'val_{key}'] = val_scores[key]

            # log metrics to wandb
            if use_wandb:
                wandb.log(metrics)
            if verb:
                print(f'[Epoch {str_epoch}]')
                beaty_utils.pprint(metrics)

            # saving best weights by loss
            if self.val_loss_meter.is_loss_best():
                checkpoint_path = '_'.join([f'[{str_epoch}]', self.cfg.save_name, 'loss', str(val_loss)])
                checkpoint_path = os.path.join(self.cfg.save_folder, checkpoint_path)
                self.remember_state_dict(checkpoint_path, epoch)

            # saving best weights by score
            if self.val_score_meter.is_score_best():
                checkpoint_path = '_'.join([f'[{str_epoch}]', self.cfg.save_name, 'score'])
                checkpoint_path = os.path.join(self.cfg.save_folder, checkpoint_path)
                self.remember_state_dict(checkpoint_path, epoch)

            # weapon counter over-fitting
            self.early_stopping.step()
            if self.early_stopping.stop_training():
                print('[!] EarlyStopping')
                break

            if epoch % 5 == 0:
                self.save_state_dict('loss')
                self.save_state_dict('score')

        self.save_state_dict('loss')
        self.save_state_dict('score')

        if use_wandb:
            wandb.finish()

        print('[X] Training is over.')

    def train_epoch(self):
        self.train_loss_meter.null()
        self.train_score_meter.null()
        self.model.train()

        for X, y in tqdm.tqdm(self.train_dl):
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X)
            self.train_score_meter.update(output > 0.5, y)
            loss = self.train_loss_meter.update(output, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def eval_epoch(self):
        self.val_loss_meter.null()
        self.val_score_meter.null()
        self.model.eval()
        for X, y in tqdm.tqdm(self.val_dl):
            X = X.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                output = self.model(X)
                self.val_loss_meter.update(output, y)
                self.val_score_meter.update(output > 0.5, y)
