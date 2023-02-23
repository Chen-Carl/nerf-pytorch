class EarlyStopping:

    def __init__(self, patience: int = 30, margin: float = 1e-4):
        # In our case PSNR
        self.best_fitness = 0.0
        self.best_iter = 0
        self.margin = margin
        # epochs to wait after fitness stops improving to stop
        self.patience = patience or float('inf')  

    def __call__(self, iter: int, fitness: float):
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop
