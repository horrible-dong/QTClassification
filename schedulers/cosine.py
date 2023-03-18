import math

from torch.optim import lr_scheduler as lr_scheduler


class CosineLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epochs, lrf):
        cosine_lr_lambda = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
        super().__init__(optimizer, lr_lambda=cosine_lr_lambda)
