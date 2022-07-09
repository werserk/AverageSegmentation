import os
import json


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        # saving
        self.save_folder = 'checkpoints/'
        self.save_name = 'model'

        # training parameters
        self.epochs = 10
        self.start_epoch = 1
        self.end_epoch = 10
        self.device = 'cuda'

        # modules for stuff
        self.metric_module = 'modeling.metrics'
        self.criterion_module = 'modeling.losses'
        self.models_module = 'modeling.models'
        self.optimizer_module = 'torch.optim'
        self.scheduler_module = 'torch.optim.lr_scheduler'

        # training stuff
        self.model = 'Unet'
        self.model_params = ...
        self.criterion = 'IoULoss'
        self.metric = 'IoUScore'
        self.optimizer = 'Adam'
        self.scheduler_module = 'OneCycleLR'

    def load(self, path=None):
        assert os.path.exists(path), f"{path} does not exist"
        with open(path) as f:
            data = json.load(f)
        for key in data.keys():
            self.__setattr__(key, data[key])
        return self

    def save(self, replace=False):
        configs_path = os.path.join(self.save_folder, 'configs')
        if not os.path.exists(configs_path):
            os.makedirs(configs_path)
            print(f"{configs_path} created successfully")
        save_path = os.path.join(configs_path, self.save_name) + '.cfg'
        if not replace:
            assert not os.path.exists(save_path), f"{save_path} already exists"
        with open(save_path, 'w') as f:
            json.dump(self, f)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
