import os
import json


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

        # base
        self.seed = 42
        self.device = 'cuda'

        # saving
        self.save_folder = 'aseg_checkpoints/'
        self.save_name = 'Unet'
        self.wandb_project = 'aseg_project'

        # loading
        self.data_folder = 'data/'

        # training parameters
        self.epochs = 10
        self.start_epoch = 1
        self.end_epoch = -1
        self.batch_size = 1
        self.split_sizes = (0.8, 0.2)
        self.stop_earlystopping_step = 2

        # augmentations
        self.train_transforms = []
        self.val_transforms = []
        self.test_transforms = []

        # modules for stuff
        self.metric_module = 'modeling.metrics'
        self.criterion_module = 'modeling.losses'
        self.models_module = 'modeling.models'
        self.optimizer_module = 'torch.optim'
        self.scheduler_module = 'torch.optim.lr_scheduler'

        # training stuff
        self.model = 'Unet'
        self.model_params = {'backbone': 'resnet101',
                             'num_classes': 2,
                             'in_channels': 3,
                             'layers_to_freeze': 2,
                             'encoder_weights': ''}
        self.criterion = 'IoULoss'
        self.criterion_params = {}
        self.metric = 'IoUScore'
        self.metric_params = {}
        self.optimizer = 'RAdam'
        self.optimizer_params = {'lr': 1e-3}
        self.scheduler = 'OneCycleLR'
        self.scheduler_params = {'max_lr': 1e-3}

    def load(self, path=None):
        assert os.path.exists(path), f"{path} does not exist"
        with open(path) as f:
            data = json.load(f)
        for key in data.keys():
            self.__setattr__(key, data[key])
        return self

    def save(self, replace=False):
        save_path = os.path.join(self.save_folder, self.save_name) + '.json'
        if not replace:
            assert not os.path.exists(save_path), f"{save_path} already exists"
        with open(save_path, 'w') as f:
            json.dump(self, f)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
