class EarlyStopping:
    def __init__(self, train_loss_meter, val_loss_meter, max_step=-1):
        self.current_step = 0
        self.train_loss_meter = train_loss_meter
        self.val_loss_meter = val_loss_meter
        self.max_step = max_step

    def step(self):
        if self.train_loss_meter.is_loss_decreasing() and not self.val_loss_meter.is_loss_decreasing():
            self.current_step += 1
        else:
            self.current_step = 0

    def stop_training(self):
        return self.current_step >= self.max_step
