from embedding import *
from collections import OrderedDict
import torch
import torch.nn.functional as F

# 100
class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(3*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=4,
            kdim=embed_size,
            vdim=embed_size
        )

        # self.m_model = Mamba(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=100,  # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # ).to("cuda")

    def forward(self, inputs, time):

        # head = inputs[:,:,0,:]
        # tail = inputs[:,:,1,:]
        #
        # query = (head * time)  # [1, batch_size, entity_dim]
        # key = tail  # [1, batch_size, entity_dim]
        # value = tail  # [1, batch_size, entity_dim]
        #
        # # 4. 执行注意力计算
        # attn_output, attn_weights = self.multihead_attn(query, key, value)
        # attn_output = attn_output  # [batch_size, entity_dim]
        #
        # # 5. 组合并投影
        # combined = torch.cat([attn_output, tail, time], dim=-1)
        #
        #
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = torch.cat((x,time),dim=-1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)


        # x = self.m_model(x)

        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)
        # return x.view(size[0], 1, 1, -1)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()
        self.epsilon = 1e-8  # 添加小量以防止除零

    def forward(self, h, t, r, tem, pos_num, norm):
        #h,r,t [1024,2,1,400]
        # norm = norm[:, :1, :, :]  # revise
        # h = h - torch.sum(h * norm, -1, True) * norm
        # t = t - torch.sum(t * norm, -1, True) * norm
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)    #[1024,2]
        p_score = score[:, :pos_num]    #[1024,1]
        n_score = score[:, pos_num:]    ##[1024,1]
        return p_score, n_score

        # h_plus_r = h + r  # [1024,2,1,400]
        #
        # # 计算余弦相似度
        # # 首先计算 L2 范数
        # h_plus_r_norm = torch.norm(h_plus_r, p=2, dim=-1, keepdim=True)  # [1024,2,1,1]
        # t_norm = torch.norm(t, p=2, dim=-1, keepdim=True)  # [1024,2,1,1]
        #
        # # 归一化向量
        # h_plus_r_normalized = h_plus_r / (h_plus_r_norm + self.epsilon)  # [1024,2,1,400]
        # t_normalized = t / (t_norm + self.epsilon)  # [1024,2,1,400]
        #
        # # 计算点积（cosine similarity）
        # score = torch.sum(h_plus_r_normalized * t_normalized, dim=-1).squeeze(2)  # [1024,2]
        #
        # # 分割正例和负例的分数
        # p_score = score[:, :pos_num]  # [1024,1]
        # n_score = score[:, pos_num:]  # [1024,1]
        #
        # return p_score, n_score


class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'ICWES18-One'or parameter['dataset'] == 'icews-few-intp-One' or parameter['dataset'] == 'gdelt-few-intp-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=600,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin, reduction='mean')
        self.margin_loss = nn.MarginRankingLoss(0.02, reduction='mean')
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2  #[1024,2,1,400]


    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        # support,support_neg [1024,5,2,400] query,negative [1024,3,2,400]
        few = support[0].shape[1]              # num of few
        num_sn = support_negative[0].shape[1]  # num of support negative
        num_q = query[0].shape[1]              # num of query
        num_n = negative[0].shape[1]           # num of query negative
        # query_t=0
        # support_t=0
        support_t = support[1].repeat(1, int((few + num_sn) / support[1].shape[1]), 1).unsqueeze(2)
        query_t = query[1].repeat(1, int((num_q + num_n) / query[1].shape[1]), 1).unsqueeze(2)

        rel = self.relation_learner(support[0],support[1])  #[1024,1,1,400]
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)  #[1024,2,1,400]

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support[0], support_negative[0]) #[1024,2,1,400]

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, support_t, few, few)
                # if not iseval:
                #     p_score = p_score.unsqueeze(2).repeat(1, 1, 2).reshape(1024, -1)
                # else:
                #     p_score = p_score.unsqueeze(2).repeat(1, 1, 2).reshape(1, -1)

                # y = torch.Tensor([1]).to(self.device)
                y = torch.ones_like(n_score).to(self.device)
                self.zero_grad()

                loss = self.loss_func(p_score, n_score, y)

                e2 = sup_neg_e1[:,:few,:,:] + rel_s[:,:few,:,:]
                pred_head_sim = torch.cosine_similarity(sup_neg_e1[:, :few, :, :], e2, dim=-1)
                head_tail_sim = torch.cosine_similarity(sup_neg_e1[:, :few, :, :], sup_neg_e2[:, :few, :, :],
                                                        dim=-1)
                pred_tail_sim = torch.cosine_similarity(sup_neg_e2[:, :few, :, :], e2, dim=-1)

                target = torch.ones_like(pred_tail_sim)

                loss1 = self.margin_loss(pred_tail_sim, pred_head_sim, target)

                loss = loss+0*loss1

                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query[0], negative[0])  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, query_t, num_q, num_q)
        # if not iseval:
        #     p_score = p_score.unsqueeze(2).repeat(1, 1, 2).reshape(1024, -1)

        return p_score, n_score, que_neg_e1, que_neg_e2, rel_q, query_t

