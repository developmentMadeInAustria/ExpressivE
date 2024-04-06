import torch
from pykeen.regularizers import Regularizer


class ExpressivERegularizerEmpty(Regularizer):

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return 0