import torch
import torch.nn as nn
import math

from mamba_ssm import Mamba



class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.es = parameter['embed_dim']
        self.tem = dataset['tem_total']

        num_ent = len(self.ent2id)
        # num_ent self.tem
        self.embedding = nn.Embedding(num_ent, self.es)
        self.tem_embeddings = nn.Embedding(self.tem, self.es)

        self.tem2emb = dataset['tem2emb']
        # nn.init.xavier_uniform_(self.embedding.weight)
        self.tem_embeddings.weight.data.copy_(torch.from_numpy(self.tem2emb))

        if parameter['data_form'] == 'Pre-Train':
            self.ent2emb = dataset['ent2emb']
            self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
            # nn.init.xavier_uniform_(self.embedding.weight)
        elif parameter['data_form'] in ['In-Train', 'Discard']:
            nn.init.xavier_uniform_(self.embedding.weight)

        # self.PE = self.generate_positional_encoding(self.es, 5000).to(self.device)

    def generate_positional_encoding(self, d_model, max_len):
        """
        Create standard transformer PEs.
        Inputs :
          d_model is a scalar correspoding to the hidden dimension
          max_len is the maximum length of the sequence
        Output :
          pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, triples):
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx_tem = [[t[3] for t in batch] for batch in triples]
        idx_tem = torch.LongTensor(idx_tem).to(self.device)
        idx = torch.LongTensor(idx).to(self.device)
        a = self.embedding(idx)
        b = self.tem_embeddings(idx_tem)
        # b=0
        # b = self.PE[idx_tem]

        return a, b
        # return a

