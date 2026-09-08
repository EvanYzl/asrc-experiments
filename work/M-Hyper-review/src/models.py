import torch
import random
from torch import nn
from tqdm import tqdm
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
from qutils import *
import os
import numpy as np


class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.inference(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        #filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out = filters.get((query[0].item(), query[1].item()), [])
                        filter_out += [queries[b_begin + i, 2].item()]   
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks
    def independent_loss(self, embed_x, embed_y):
        """ Minimize mutual information by making embeddings less similar. """
        # Compute similarity score (e.g., cosine similarity) for the joint embedding
        positive_similarity = F.cosine_similarity(embed_x, embed_y, dim=1)
        # Generate negative samples by shuffling
        embed_y_shuffled = embed_y[torch.randperm(embed_y.size(0))]
        # Compute similarity score for the shuffled embedding
        negative_similarity = F.cosine_similarity(embed_x, embed_y_shuffled, dim=1)
        # Loss is the difference between negative and positive similarity
        # Larger negative similarity implies less mutual information
        loss = torch.mean(negative_similarity) - torch.mean(positive_similarity)
        return loss
    
class generate_sparse_embeddings(nn.Module):
    def __init__(self, preserve_ratio=0.2):
        super(generate_sparse_embeddings, self).__init__()
        self.preserve_ratio = preserve_ratio
    def forward(self, num_ent, dim, mean, std):
        """
        优化后的稀疏嵌入生成：
        只生成需要保留行的嵌入，然后将它们填充到稀疏矩阵。
        """
        # 计算需要保留的行数
        num_preserve = int(num_ent * self.preserve_ratio)
        # 随机选取要生成嵌入的行索引
        selected_indices = torch.randperm(num_ent)[:num_preserve]
        # 符号位：生成多行正态分布值，仅为保留行生成嵌入
        selected_embeddings = torch.normal(mean=mean.expand((num_preserve, dim)), std=std.expand((num_preserve, dim)))  # Shape: [num_preserve, dim]
        # 创建一个稀疏矩阵，初始为全零
        sparse_embeddings = torch.zeros(num_ent, dim,device=std.device)
        # 填充保留的行到稀疏矩阵
        sparse_embeddings[selected_indices] = selected_embeddings
        return sparse_embeddings

class rel_fusion(nn.Module):
    def __init__(self, rank: int, num_rel:int):
        super(rel_fusion, self).__init__()
        self.fc_s = nn.Linear(rank * 18, 1)
        self.fc_v = nn.Linear(rank * 18, 1)
        self.fc_t = nn.Linear(rank * 18, 1)
        self.ids_r = nn.Parameter(torch.ones(num_rel))

    def forward(self, E_s, E_v, E_t, E_r, id_r):
        score_s = self.fc_s(torch.cat((E_s, E_r), dim=-1)) 
        score_v = self.fc_v(torch.cat((E_v, E_r), dim=-1))
        score_t = self.fc_t(torch.cat((E_t, E_r), dim=-1))
        temperature = self.ids_r[id_r].view(-1, 1)  # (batch_size, 1)

        scores = torch.cat((score_s, score_v, score_t), dim=-1)  # (batch_size, 3)
        weights = F.softmax(scores/temperature, dim=-1)  # (batch_size, 3)

        # Compute weighted sum of embeddings
        fused_embedding = weights[:, 0:1] * E_s + weights[:, 1:2] * E_v + weights[:, 2:3] * E_t

        return fused_embedding

class FERF(nn.Module):
    def __init__(self, rank: int, img_emb_size, text_emb_size):
        super(FERF, self).__init__()
        self.init_size = 1e-3,
        self.fc_s = nn.Linear(rank * 6, rank * 2)
        self.fc_v = nn.Linear(rank * 6, rank * 2)#img_emb_size)
        self.fc_t = nn.Linear(rank * 6, rank * 2)#text_emb_size)


    def forward(self, E_s2, E_s1, E_v2, E_v1, E_t2, E_t1):
        temp_s = self.fc_s(torch.cat((E_s2, E_v1, E_t1), dim=-1)) 
        temp_v = self.fc_v(torch.cat((E_s1, E_v2, E_t1), dim=-1))
        temp_t = self.fc_t(torch.cat((E_s1, E_v1, E_t2), dim=-1))
    
        return E_s1+E_s2, E_v1+E_v2, E_t1+E_t2, temp_s, temp_v, temp_t
        #return final_s, final_v, final_t, loss_recon1+loss_recon2
        #return E_s1+E_s2, E_v1+E_v2, E_t1+E_t2, 0


class M_Hyper_B(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, # num_entities, num_relations*2, num_entities
            init_size: float = 1e-3,
            img_emb = None, text_emb = None
    ):
        super(M_Hyper_B, self).__init__()
        self.all = nn.Embedding(sizes[0], 2 * rank, sparse=True)
        self.all.weight.data *= init_size

        self.structure = nn.Embedding(sizes[0], 2 * rank, sparse=True)
        self.structure.weight.data *= init_size
        self.stru = nn.Embedding(sizes[0], 2 * rank, sparse=True)
        self.stru.weight.data *= init_size
        self.stru_mean = self.stru.weight.data.mean(dim=0, keepdim=True) #[1,dim]
        self.stru_mean.data *= init_size
        self.stru_std = self.stru.weight.data.std(dim=0, keepdim=True) #[1,dim]
        self.stru_std.data *= init_size

        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.img_embeddings.weight.data *= init_size
        self.img_mean = self.img_embeddings.weight.data.mean(dim=0, keepdim=True) #[1,dim]
        self.img_std = self.img_embeddings.weight.data.std(dim=0, keepdim=True) #[1,dim]

        # pca initial self.img
        pca = PCA(n_components=2 * rank)
        reduced_weights = torch.tensor(pca.fit_transform(self.img_embeddings.weight.data.numpy()), dtype=torch.float32, device=self.img_embeddings.weight.device)
        self.img = nn.Embedding(sizes[0], 2 * rank, sparse=True)
        self.img.weight.data.copy_(reduced_weights)
        self.img.weight.data *= init_size
        self.img_proj = nn.Linear(img_emb.size()[1], rank*2)

        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(False)
        self.text_embeddings.weight.data *= init_size
        self.text_mean = self.text_embeddings.weight.data.mean(dim=0, keepdim=True) #[1,dim]
        self.text_std = self.text_embeddings.weight.data.std(dim=0, keepdim=True) #[1,dim]

        # pca initial self.text
        reduced_weights = torch.tensor(pca.fit_transform(self.text_embeddings.weight.data.numpy()), dtype=torch.float32, device=self.text_embeddings.weight.device)
        self.text = nn.Embedding(sizes[0], 2 * rank, sparse=True)
        self.text.weight.data.copy_(reduced_weights)
        self.text.weight.data *= init_size
        self.text_proj = nn.Linear(text_emb.size()[1], rank*2)
        
        self.FERF = FERF(rank,img_emb.size()[1],text_emb.size()[1])
        self.rel_fusion = rel_fusion(rank, sizes[1])
        self.generate_sparse_embeddings = generate_sparse_embeddings(0.2)

        self.sizes = sizes
        self.rank = rank
        self.rel_embedding = nn.Embedding(sizes[1], 16 * rank, sparse=True)
        self.rel_embedding.weight.data *= init_size


    def forward(self, x):
        rel = self.rel_embedding(x[:, 1]) # r

        device = self.img_embeddings.weight.device
        #add noise
        stru_embeddings = self.structure.weight + self.generate_sparse_embeddings(self.sizes[0], self.structure.weight.size()[1], mean=self.stru_mean.to(device), std=self.stru_std.to(device))
        img_embeddings = self.img_embeddings.weight + self.generate_sparse_embeddings(self.sizes[0], self.img_embeddings.weight.size()[1], mean=self.img_mean.to(device), std=self.img_std.to(device))
        text_embeddings = self.text_embeddings.weight + self.generate_sparse_embeddings(self.sizes[0], self.text_embeddings.weight.size()[1], mean=self.text_mean.to(device), std=self.text_std.to(device))
        proj_noise_stru, proj_noise_img, proj_noise_text = stru_embeddings, self.img_proj(img_embeddings), self.text_proj(text_embeddings)
        stru, image, text, recon_stru1, recon_img1, recon_text1 = self.FERF(self.stru.weight, proj_noise_stru, self.img.weight, proj_noise_img, self.text.weight, proj_noise_text)
 
        # without noise
        proj_no_stru, proj_no_img, proj_no_text = self.structure.weight, self.img_proj(self.img_embeddings.weight), self.text_proj(self.text_embeddings.weight)
        stru_no_noise, image_no_noise, text_no_noise, recon_stru2, recon_img2, recon_text2 = self.FERF(self.stru.weight, proj_no_stru, self.img.weight, proj_no_img, self.text.weight, proj_no_text)
        #recon_loss = torch.nn.functional.mse_loss(torch.cat((recon_stru1,recon_stru2,recon_img1,recon_img2,recon_text1,recon_text2),dim=-1), torch.cat((stru_embeddings,self.structure.weight, img_embeddings,self.img_embeddings.weight, text_embeddings,self.text_embeddings.weight),dim=-1), reduction='mean')
        recon_loss = torch.nn.functional.mse_loss(torch.cat((recon_stru1,recon_stru2,recon_img1,recon_img2,recon_text1,recon_text2),dim=-1), torch.cat((proj_noise_stru,proj_no_stru,proj_noise_img,proj_no_img,proj_noise_text,proj_no_stru),dim=-1), reduction='mean')
        
        #all = self.all.weight
        all_no_noise = self.rel_fusion(stru_no_noise[x[:, 0]], image_no_noise[x[:, 0]], text_no_noise[x[:, 0]], rel,x[:, 1])  # [batch_size,8*rank]
        all_no_noise += self.all.weight[x[:, 0]]
        all = self.rel_fusion(stru[x[:, 0]], image[x[:, 0]], text[x[:, 0]], rel,x[:, 1])  # [batch_size,8*rank]
        all += self.all.weight[x[:, 0]]
        # self-distillation loss
        consistency_loss = torch.nn.functional.mse_loss(all_no_noise, all, reduction='mean')
        # mutual information loss
        mut_loss1 = self.independent_loss(proj_noise_stru,self.stru.weight) + self.independent_loss(proj_no_stru,self.stru.weight)
        mut_loss2 = self.independent_loss(proj_noise_img,self.img.weight) + self.independent_loss(proj_no_img,self.img.weight)
        mut_loss3 = self.independent_loss(proj_noise_text,self.text.weight) + self.independent_loss(proj_no_text,self.text.weight)

        self.ent_embedding = torch.cat([self.all.weight, stru, image, text], dim=-1) # [num_ent, 8*rank] 
        
        lhs = torch.cat([all, stru[x[:, 0]], image[x[:, 0]], text[x[:, 0]]], dim=-1) # h  [batch_size, 8*rank] 
        rhs = self.ent_embedding[x[:, 2]] # t

        # print(lhs.size())
        # torch.save(lhs.cpu(), f'../case/country_{self.n}.pth')
        # self.n=self.n+1

        lhs += rel[:, self.rank * 8:]  # Q_{h,r}
        w_a, x_a, y_a, z_a = torch.split(lhs, self.rank * 2, dim=-1)
        w_b, x_b, y_b, z_b = torch.split(rel[:, :self.rank*8], self.rank * 2, dim=-1)

        # A = complex_mul(w_a,w_b) - complex_mul(x_a,x_b) - complex_mul(y_a,y_b) - complex_mul(z_a,z_b)  
        # B = complex_mul(w_a,x_b) + complex_mul(x_a,w_b) + complex_mul(y_a,z_b) - complex_mul(z_a,y_b)  
        # C = complex_mul(w_a,y_b) - complex_mul(x_a,z_b) + complex_mul(y_a,w_b) + complex_mul(z_a,x_b)  
        # D = complex_mul(w_a,z_b) + complex_mul(x_a,y_b) - complex_mul(y_a,x_b) + complex_mul(z_a,w_b) 
        # A = complex_mul(w_a,w_b) # ensemble
        # B = complex_mul(x_a,x_b)
        # C = complex_mul(y_a,y_b)
        # D = complex_mul(z_a,z_b) 

        # res = torch.cat([A, B, C, D], dim=-1)
        # fusion
        res = complex_mul((torch.cat([w_a,x_a,y_a,z_a], dim=-1)),(torch.cat([w_b,x_b,y_b,z_b], dim=-1)))
        return  res @ self.ent_embedding.transpose(0, 1), [(get_norm(lhs, 8), get_norm(rel[:, :self.rank*8], 8), get_norm(rhs, 8))], consistency_loss+recon_loss#+mut_loss1+mut_loss2+mut_loss3

    def inference(self, x):
        rel = self.rel_embedding(x[:, 1]) # r

        device = self.img_embeddings.weight.device
        #without noise
        stru_embeddings = self.structure.weight #+ self.generate_sparse_embeddings(self.sizes[0], self.structure.weight.size()[1], mean=self.stru_mean.to(device), std=self.stru_std.to(device))
        img_embeddings = self.img_embeddings.weight #+ self.generate_sparse_embeddings(self.sizes[0], self.img_embeddings.weight.size()[1], mean=self.img_mean.to(device), std=self.img_std.to(device))
        text_embeddings = self.text_embeddings.weight #+ self.generate_sparse_embeddings(self.sizes[0], self.text_embeddings.weight.size()[1], mean=self.text_mean.to(device), std=self.text_std.to(device))
        
        stru, image, text,_ ,_ ,_ = self.FERF(self.stru.weight, stru_embeddings, self.img.weight, self.img_proj(img_embeddings), self.text.weight, self.text_proj(text_embeddings))

        #all = self.all.weight
        all = self.rel_fusion(stru[x[:, 0]], image[x[:, 0]], text[x[:, 0]],rel,x[:, 1])  # [batch_size,8*rank]
        all += self.all.weight[x[:, 0]]

        self.ent_embedding = torch.cat([self.all.weight, stru, image, text], dim=-1) # [num_ent, 8*rank]

        lhs = torch.cat([all, stru[x[:, 0]], image[x[:, 0]], text[x[:, 0]]], dim=-1) # h  [batch_size, 8*rank] 
        rhs = self.ent_embedding[x[:, 2]] # t

        lhs += rel[:, self.rank * 8:]  # Q_{h,r}
        w_a, x_a, y_a, z_a = torch.split(lhs, self.rank * 2, dim=-1)
        w_b, x_b, y_b, z_b = torch.split(rel[:, :self.rank*8], self.rank * 2, dim=-1)

        # A = complex_mul(w_a,w_b) - complex_mul(x_a,x_b) - complex_mul(y_a,y_b) - complex_mul(z_a,z_b)  
        # B = complex_mul(w_a,x_b) + complex_mul(x_a,w_b) + complex_mul(y_a,z_b) - complex_mul(z_a,y_b)  
        # C = complex_mul(w_a,y_b) - complex_mul(x_a,z_b) + complex_mul(y_a,w_b) + complex_mul(z_a,x_b)  
        # D = complex_mul(w_a,z_b) + complex_mul(x_a,y_b) - complex_mul(y_a,x_b) + complex_mul(z_a,w_b) 
        # A = complex_mul(w_a,w_b) # ensemble
        # B = complex_mul(x_a,x_b)
        # C = complex_mul(y_a,y_b)
        # D = complex_mul(z_a,z_b)

        #res = torch.cat([A, B, C, D], dim=-1)
        res = complex_mul((torch.cat([w_a,x_a,y_a,z_a], dim=-1)),(torch.cat([w_b,x_b,y_b,z_b], dim=-1)))
        return  res @ self.ent_embedding.transpose(0, 1), [(get_norm(lhs, 8), get_norm(rel[:, :self.rank*8], 8), get_norm(rhs, 8))]

