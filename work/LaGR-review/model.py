from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from lightning import LightningModule

from metric import KGCMetric
from backbones import RelationalGraphNetwork, GatedAttentionLayer


class FusionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int,
                 graph_layers: int, graph_intermediate_size: int, graph_dropout: float, graph_norm: str,
                 attn_heads: int, attn_dropout: float, attn_intermediate_size: int, attn_leaders: int, attn_norm: str):
        super().__init__()
        self.graph = RelationalGraphNetwork(num_relations, graph_layers, hidden_size,
                                            graph_intermediate_size, graph_dropout, graph_norm)
        self.attn = GatedAttentionLayer(hidden_size, attn_heads, attn_dropout, attn_intermediate_size,
                                        attn_leaders, attn_norm)
        self.w_proj = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_size, hidden_size))

        self.x_proj = nn.Linear(hidden_size, hidden_size)
        self.v_emb = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.normal_(self.v_emb)
        self.xv_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, edge_index, edge_type, z, x, v):
        x = self.x_proj(x) + v * self.v_emb
        x = self.xv_proj(F.leaky_relu(x))

        a = torch.jit.fork(self.graph, edge_index, edge_type, z, x)
        b = torch.jit.fork(self.attn, x)
        a, b = torch.jit.wait(a), torch.jit.wait(b)

        return self.w_proj(torch.cat((a, b), dim=-1)) + a


class FusionModel(LightningModule):
    r"""LaGR

        Args:
            graph_intermediate_size (int): intermediate size of the feed-forward network in GNNs.
            attn_intermediate_size (int): intermediate size of the feed-forward network in global attention modules.
            num_blocks (int): number of LaGR blocks.
            graph_layers (int): number of GNN layers in each block.
            graph_dropout (float): dropout rate of the feed-forward network in GNNs.
            graph_norm (str): norm class using in GNNs. Support classes ['LN', 'TANH'].
            attn_heads (int): number of attention heads in global attention modules.
            attn_leaders(int): size of latent space in global attention modules.
            attn_dropout (float): dropout rate of the feed-forward network in global attention modules.
            attn_norm (str): norm class using in global attention modules.
            hits (str): hits metrics, e.g., 0,1,10 for using MRR, hits@1 and hits@10. Default is 0,1,3,10.
            remove_all (bool): removing training samples from adjacency matrix. For the triplet (h, r, t), remove (h, *, t) if True. Otherwise, remove (h, r, t) only.
            loss_fn (str): Loss function. Support arguments ['bce', 'ce', 'mce', Any], referring to binary cross entropy (with negative samples), cross entropy (without negative samples), cross entropy (with negative samples), and binary cross entropy (without negative samples).
            num_negative_sample (int): For loss function 'bce' only.
            adversarial_temperature (int): For loss function 'bce' only.
            top_k (int): Coarse-to-fine optimization argument k.
            delta (int): Coarse-to-fine optimization argument delta.
    """

    MAIN_METRIC = 'valid_MRR'

    def __init__(self, num_relations: int,
                 hidden_size: int = 32, graph_intermediate_size: int = 32, attn_intermediate_size: int = 128,
                 num_blocks: int = 3, graph_layers: int = 3, graph_dropout: float = 0.2, graph_norm: str = 'LN',
                 attn_heads: int = 4, attn_leaders: int = 200, attn_dropout: float = 0.1, attn_norm: str = 'LN',
                 num_nodes: Optional[int] = None, max_epochs: Optional[int] = None, hits: Optional[str] = None,
                 optimizer: str = 'Adam', learning_rate: float = 5e-3, weight_decay: float = 1e-4,
                 use_scheduler: bool = True, remove_all: bool = False, loss_fn: str = 'bce',
                 num_negative_sample: Optional[int] = None, adversarial_temperature: Optional[float] = None,
                 top_k: int = 4, delta: int = 8):
        super().__init__()
        self.save_hyperparameters()

        self.query_embedding = nn.Embedding(num_relations, hidden_size)
        block = FusionBlock(hidden_size, num_relations, graph_layers, graph_intermediate_size, graph_dropout,
                            graph_norm, attn_heads, attn_dropout, attn_intermediate_size, attn_leaders, attn_norm)
        self.encoder = nn.ModuleList([deepcopy(block) for _ in range(num_blocks)])
        self.mlp_out = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_size, 1))
        self.node_features = None
        if num_nodes is not None:
            self.node_features = nn.Parameter(torch.empty((1, num_nodes, hidden_size)), requires_grad=False)
            nn.init.normal_(self.node_features)

        self.metric_keys = [0, 1, 3, 10] if hits is None else [int(i) for i in hits.split(',')]
        for i in self.metric_keys:
            setattr(self, f'metric_raw_{i}', KGCMetric(i))
            setattr(self, f'metric_refine_{i}', KGCMetric(i))

    def _encode(self, batch):
        h_index, r_index = batch['data'][:, 0], batch['data'][:, 1]
        edge_index, edge_type = batch['edge_index'], batch['edge_type']
        num_nodes = batch['num_nodes']
        batch_size = h_index.size(0)

        z = self.query_embedding(r_index)
        if self.node_features is None:
            x = torch.zeros((batch_size, num_nodes, z.size(-1)), device=z.device)
        else:
            x = self.node_features.repeat(batch_size, 1, 1)
        v = torch.zeros_like(x)
        v[torch.arange(batch_size, device=z.device), h_index] = 1

        for layer in self.encoder:
            x = layer(edge_index, edge_type, z, x, v)

        out = self.mlp_out(x).squeeze(-1)
        return out

    def training_step(self, batch):
        h_index, r_index, t_index = batch['data'].unbind(-1)
        num_nodes = batch['num_nodes']
        h = torch.cat([h_index, t_index], 0)
        t = torch.cat([t_index, h_index], 0)
        if self.hparams.remove_all:
            encode_fn = lambda x, _, z: x + z * num_nodes
            r = None
        else:
            encode_fn = lambda x, y, z: z + (x + y * num_nodes) * num_nodes
            r = torch.cat([r_index, torch.where(r_index % 2 == 0, r_index + 1, r_index - 1)], 0)

        edge_index, edge_type = batch['edge_index'], batch['edge_type']
        mask = ~torch.isin(encode_fn(edge_index[0, :], edge_type, edge_index[1, :]), encode_fn(h, r, t))
        batch['edge_index'] = edge_index[:, mask]
        batch['edge_type'] = edge_type[mask]

        score = self._encode(batch)
        if self.hparams.loss_fn == 'ce':
            loss = F.cross_entropy(score, t_index)
        elif self.hparams.loss_fn == 'mce':
            mask = batch['mask']
            mask[torch.arange(score.shape[0], device=score.device), t_index] = 1
            score.masked_fill_(~mask.bool(), -1e4)
            loss = F.cross_entropy(score, t_index)
        else:
            negative_index = torch.multinomial(batch['mask'], replacement=True,
                                               num_samples=self.hparams.num_negative_sample)
            logits = torch.gather(score, 1, torch.cat([t_index.unsqueeze(1), negative_index], 1))
            if self.hparams.loss_fn == 'bce':
                target = torch.zeros_like(logits)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
                weights = torch.ones_like(logits)
                with torch.no_grad():
                    weights[:, 1:] = F.softmax(logits[:, 1:] / self.hparams.adversarial_temperature, dim=-1)
                loss = (loss * weights).sum()
            else:
                loss = F.cross_entropy(logits, torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64))

        self.log('loss', loss.item(), prog_bar=True)
        return loss

    def _call_metric(self, split, func, *args, **kwargs):
        results = {}
        for i in self.metric_keys:
            metric = getattr(self, f'metric_{split}_{i}')
            results[metric.name] = getattr(metric, func)(*args, **kwargs)
        return results

    def validation_step(self, batch):
        scores = self._encode(batch)
        answer_scores = scores.gather(1, batch['data'][:, [-1]])
        ranks = torch.sum((scores >= answer_scores) & batch['mask'].bool(), dim=1) + 1
        self._call_metric('raw', 'update', ranks)

    def test_step(self, batch):
        scores = self._encode(batch)
        answers = batch['data'][:, [-1]]
        answer_scores = scores.gather(1, answers)
        masks = batch['mask'].bool()
        ranks = torch.sum((scores >= answer_scores) & masks, dim=1) + 1
        self._call_metric('raw', 'update', ranks)

        if batch['coarse'] is None:
            return

        coarse_indices = batch['coarse']
        top_k, delta = self.hparams.top_k, self.hparams.delta

        top_indices, bot_indices = coarse_indices[:, :top_k], coarse_indices[:, top_k:]
        top_scores = torch.gather(scores, 1, top_indices)
        bot_scores = torch.gather(scores, 1, bot_indices)
        diff = bot_scores.max(dim=1).values - top_scores.max(dim=1).values
        is_in_top = (answers == top_indices).any(dim=1)

        condition1 = (diff <= delta) & is_in_top
        condition2 = (diff > delta) & (~is_in_top)
        ranks = torch.where(
            condition1,
            ((top_scores >= answer_scores) & masks.gather(1, top_indices)).sum(dim=1) + 1,
            torch.where(
                condition2,
                ((bot_scores >= answer_scores) & masks.gather(1, bot_indices)).sum(dim=1) + 1,
                ((scores >= answer_scores) & masks).sum(dim=1) + 1
            )
        )
        self._call_metric('refine', 'update', ranks)

    def on_validation_epoch_end(self):
        for k, v in self._call_metric('raw', 'compute').items():
            self.log(f'valid_{k}', v, prog_bar=True, sync_dist=True)
        self._call_metric('raw', 'reset')

    def on_test_epoch_end(self):
        for k, v in self._call_metric('raw', 'compute').items():
            self.log(f'raw_{k}', v, prog_bar=False, sync_dist=True)
        self._call_metric('raw', 'reset')
        for k, v in self._call_metric('refine', 'compute').items():
            self.log(f'refine_{k}', v, prog_bar=False, sync_dist=True)
        self._call_metric('refine', 'reset')

    def configure_optimizers(self):
        no_decay_list = ['bias', 'norm.weight']
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.endswith(skip_name) for skip_name in no_decay_list):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optim_groups = [
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            },
            {
                'params': decay_params,
                'weight_decay': self.hparams.weight_decay
            }
        ]
        optimizer = getattr(torch.optim, self.hparams.optimizer)(optim_groups, lr=self.hparams.learning_rate)
        if not self.hparams.use_scheduler:
            return optimizer
        n = self.hparams.max_epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(n / 2), int(n * 3 / 4)], 0.1)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
