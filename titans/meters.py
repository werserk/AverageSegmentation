import utils.cfgtools as cfg_utils


def is_number(x):
    try:
        int(x)
        return True
    except:
        return False


class ScoreMeter:
    def __init__(self, cfg):
        functions = [cfg_utils.get_metric(cfg)]
        self.functions = functions
        self.stats = {function.__name__: 0 for function in self.functions}
        self.best_mean_stats = self.stats.copy()
        self.k = 0

    def update(self, y_pred, y_true):
        for function in self.functions:
            self.stats[function.__name__] += function(y_pred, y_true)
        self.k += 1

    def is_score_best(self, mode='any'):
        assert mode in ('any', 'all') or is_number(mode), f'Not correct mode "{mode}"'
        keys = list(self.stats.keys())
        best_count = 0
        mean_stats = self.get_mean_stats()

        for key in keys:
            if mean_stats[key] > self.best_mean_stats[key]:
                best_count += 1

        if mode == 'any':
            if best_count != 0:
                self.best_mean_stats = mean_stats
                return True
            return False
        elif mode == 'all':
            if best_count == len(keys):
                self.best_mean_stats = mean_stats
                return True
            return False
        else:
            if best_count >= mode:
                self.best_mean_stats = mean_stats
                return True
            return False

    def get_mean_stats(self):
        return {key: self.stats[key] / self.k for key in list(self.stats.keys())}

    def null(self):
        self.stats = {function.__name__: 0 for function in self.functions}
        self.k = 0


class LossMeter:
    def __init__(self, cfg):
        self.criterion = cfg_utils.get_criterion(cfg)
        self.last_loss = -1
        self.best_loss = 0
        self.loss = 0
        self.k = 0

    def update(self, y_pred, y_true):
        current_loss = self.criterion(y_pred, y_true)
        self.loss += current_loss.item()
        self.k += 1
        return current_loss

    def is_loss_best(self):
        loss = self.get_mean_loss()
        if loss > self.best_loss:
            self.best_loss = loss
            return True
        return False

    def get_mean_loss(self):
        return self.loss / self.k

    def null(self):
        self.last_loss = self.loss
        self.loss = 0
        self.k = 0
