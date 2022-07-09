import torch
import wandb
import tqdm
import utils


class Trainer:
    def __init__(self, cfg, get_data_loaders=None):
        self.cfg = cfg
        utils.set_seed(self.cfg.seed)

        self.resume = False
        if get_data_loaders is None:
            get_data_loaders = utils.get_loaders

        self.device = torch.device(self.cfg.device)
        self.train_dl, self.val_dl = get_data_loaders(self.cfg)
        self.model, self.optimizer, self.scheduler, self.metric, self.criterion = utils.get_setup(self.cfg)
        self.start_epoch = 1
        try:
            self.end_epoch = self.cfg.end_epoch
        except AttributeError:
            self.end_epoch = self.cfg.epochs

    def load_state_dict(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.resume = True
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def save_state_dict(self, checkpoint_path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, checkpoint_path)

    def main_loop(self, use_wandb):
        if use_wandb:
            wandb.init(project=self.cfg.wandb_project, config=self.cfg, name=self.cfg.save_name)
            wandb.watch(self.model, log_freq=100)
            print('[*] wandb is active')

        best_val_loss = 1e3
        best_val_score = 0
        last_train_loss = 0
        last_val_loss = 1e3
        early_stopping_flag = 0
        best_state_dict = self.model.state_dict()

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            # <<<<< TRAIN >>>>>
            train_loss, train_score = self.train_epoch()

            # <<<<< EVAL >>>>>
            val_loss, val_score = self.eval_epoch()
            metrics = {'train_score': train_score,
                       'train_loss': train_loss,
                       'val_score': val_score,
                       'val_loss': val_loss,
                       'lr': self.scheduler.get_last_lr()[-1]}

            # log metrics to wandb
            if use_wandb:
                wandb.log(metrics)

            # saving best weights by loss
            if val_loss < best_val_loss:
                checkpoint_path = '_'.join([self.cfg.save_name, 'loss', str(val_loss)])
                self.save_state_dict(checkpoint_path, epoch)

            # saving best weights by score
            if val_score > best_val_score:
                checkpoint_path = '_'.join([self.cfg.save_name, 'score', str(val_loss)])
                self.save_state_dict(checkpoint_path, epoch)

            # weapon counter over-fitting
            if train_loss < last_train_loss and val_loss > last_val_loss:
                early_stopping_flag += 1
            if early_stopping_flag == self.cfg.max_early_stopping:
                print('[X] EarlyStopping')
                break

            last_train_loss = train_loss
            last_val_loss = val_loss

        if use_wandb:
            wandb.finish()

        return self.model

    def train_epoch(self):
        self.model.train()
        loss_sum = 0
        score_sum = 0
        for X, y in tqdm.tqdm(self.train_dl):
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss = loss.item()
            score = self.metric(output > 0.5, y).mean().item()
            loss_sum += loss
            score_sum += score
        return loss_sum / len(self.train_dl), score_sum / len(self.train_dl)

    def eval_epoch(self):
        self.model.eval()
        loss_sum = 0
        score_sum = 0
        for X, y in tqdm.tqdm(self.val_dl):
            X = X.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                output = self.model(X)
                loss = self.criterion(output, y).item()
                score = self.metric(output > self.cfg.threshold, y).mean().item()
                loss_sum += loss
                score_sum += score
        return loss_sum / len(self.val_dl), score_sum / len(self.val_dl)
