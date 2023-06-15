# ********************************************
# ********  QTClassification Toolkit  ********
# ********************************************
# Copyright (c) QIU, Tian. All rights reserved.

__version__ = "v0.6.0-dev"
__git_url__ = "https://github.com/horrible-dong/QTClassification"

from .criterions import build_criterion
from .datasets import build_dataset
from .evaluators import build_evaluator
from .models import build_model
from .optimizers import build_optimizer
from .schedulers import build_scheduler
