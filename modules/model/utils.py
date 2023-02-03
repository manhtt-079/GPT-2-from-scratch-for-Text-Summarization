class EarlyStopping:
    def __init__(self, patience=5, delta=1e-5):
        self.patience = patience
        self.max_save_after_not_improve = 3
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_save = True
        self.delta = delta

    def __call__(self, val_loss: float):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter <= self.max_save_after_not_improve:
                self.is_save = True
            else:
                self.is_save = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.is_save = True

