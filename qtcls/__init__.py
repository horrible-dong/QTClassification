# ********************************************
# ********  QTClassification Toolkit  ********
# ********************************************
# Copyright (c) QIU Tian. All rights reserved.

__version__ = "v0.9.1"
__git_url__ = "https://github.com/horrible-dong/QTClassification"

from .criterions import build_criterion
from .datasets import build_dataset
from .evaluators import build_evaluator
from .models import build_model
from .optimizers import build_optimizer
from .schedulers import build_scheduler


def _get_info() -> str:
    __name_version__ = f'QTClassification Toolkit {__version__}'
    max_len = max(len(__name_version__), len(__git_url__))
    edge = '*' * int(max_len * 0.15 + 1)
    return f"\n" \
           f"{edge}**{'*' * max_len}**{edge}\n" \
           f"{edge}  {__name_version__.center(max_len)}  {edge}\n" \
           f"{edge}  {__git_url__.center(max_len)}  {edge}\n" \
           f"{edge}**{'*' * max_len}**{edge}\n"


__info__ = _get_info()
