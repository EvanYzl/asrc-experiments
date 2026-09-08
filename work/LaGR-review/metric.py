import torch
from torchmetrics import Metric


class KGCMetric(Metric):
    is_differentiable: None
    higher_is_better: True
    full_state_update: True

    def __init__(self, top_k):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.top_k = top_k
        self.name = f'Hits@{top_k}' if top_k else 'MRR'

    def update(self, ranks):
        self.total += ranks.size(0)
        if self.top_k:
            self.rank_sum += (ranks <= self.top_k).sum()
        else:
            self.rank_sum += (1./ ranks).sum()

    def compute(self):
        return self.rank_sum / self.total
