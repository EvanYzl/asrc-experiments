import tqdm
import torch
from torch import nn
from torch import optim
from models import KBCModel
from regularizers import Regularizer


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    def forward(self, logits, target):
        """
        logits: 模型未归一化的输出 (batch_size, num_classes)
        target: 标签 (batch_size)，为类别索引 (0 ~ num_classes-1)
        """
        num_classes = logits.size(-1)  # 类别数
        # 找到正确类别的logit值的 log-softmax
        log_probs = torch.log_softmax(logits, dim=-1)
        # 创建平滑标签分布
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)  # (batch_size, num_classes)
            true_dist.fill_(self.smoothing / (num_classes - 1))  # 非正确类别概率
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)  # 正确类别概率
        # 计算平滑标签分布的负对数似然
        loss = -torch.sum(true_dist * log_probs, dim=-1)  # 每个样本的损失
        return loss.mean()  # 平均损失

class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256, verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        # self.LabelSmoothingCrossEntropy = LabelSmoothingCrossEntropy(smoothing=0.1)

        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors, l_con = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg + l_con

                self.optimizer.zero_grad()
                l.backward()

                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.1f}', reg=f'{l_reg.item():.1f}')

        return l
