from mBERT.Param import MbertParams
from transformers import BertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic_Bert_Unit_model(nn.Module):
    def __init__(self,input_size,result_size,MODEL_PATH):
        super(Basic_Bert_Unit_model,self).__init__()
        print(">>>>", MODEL_PATH)
        self.result_size = result_size
        self.input_size = input_size
        self.bert_model = BertModel.from_pretrained(MODEL_PATH)
        print("<<<<", MODEL_PATH)
        self.out_linear_layer = nn.Linear(self.input_size,self.result_size)  

    def forward(self,batch_word_list,attention_mask):
        x = self.bert_model(input_ids = batch_word_list,attention_mask = attention_mask)#token_type_ids =token_type_ids
        sequence_output, pooled_output = x
        cls_vec = sequence_output[:,0]        
        output = self.out_linear_layer(cls_vec)
        return output
    

def ent2Tokens_gene(Tokenizer,ent_list,ent_name_max_length =30):
    ent2tokenids = dict()
    for ent_name in ent_list:        
        token_ids = Tokenizer.encode(ent_name)[:ent_name_max_length]
        ent2tokenids[ent_name] = token_ids
    return ent2tokenids

def ent2bert_input(ent_ids,Tokenizer,ent2token_ids,des_max_length=32):
    ent2data = dict()
    pad_id = Tokenizer.pad_token_id

    for ent_name in ent_ids:
        ent2data[ent_name] = [[],[]]
        ent_token_id = ent2token_ids[ent_name]
        ent_token_ids = Tokenizer.build_inputs_with_special_tokens(ent_token_id)

        token_length = len(ent_token_ids)
        assert token_length <= des_max_length

        ent_token_ids = ent_token_ids + [pad_id] * max(0, des_max_length - token_length)

        ent_mask_ids = np.ones(np.array(ent_token_ids).shape)
        ent_mask_ids[np.array(ent_token_ids) == pad_id] = 0
        ent_mask_ids = ent_mask_ids.tolist()

        ent2data[ent_name][0] = ent_token_ids
        ent2data[ent_name][1] = ent_mask_ids
    return ent2data

def entlist2emb(Model,entids,entid2data):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]        
        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)    
    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(MbertParams.CUDA_NUM)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(MbertParams.CUDA_NUM)

    batch_emb = Model(batch_token_ids,batch_mask_ids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb

def get_embeddings(Model,ent,ent2data):
    emb = []
    batch_size=16
    for i in range(0,len(ent),batch_size):    
        batch_ents_1 = ent[i:i+batch_size]
        batch_emb_1 = entlist2emb(Model,batch_ents_1,ent2data).detach().cpu().tolist()
        emb.extend(batch_emb_1)
        del batch_emb_1
    return emb

def cos_sim_mat_generate(emb1,emb2,bs = 128):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2,dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2,dim=1)
    res_mat = batch_mat_mm(array_emb1,array_emb2.t(), bs=bs)
    return res_mat

def batch_mat_mm(mat1,mat2, bs=128):
    #be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat_1 = mat1[i:min(i+bs,axis_0)].cuda(MbertParams.CUDA_NUM)
        res = temp_div_mat_1.mm(mat2.cuda(MbertParams.CUDA_NUM))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat,0)
    return res_mat

def batch_topk(mat,bs=128,topn = 1,largest = False):
    #be equal to topk, Speed up computing with GPU
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat = mat[i:min(i+bs,axis_0)].cuda(MbertParams.CUDA_NUM)
        score_mat,index_mat =temp_div_mat.topk(topn,largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score,0)
    res_index = torch.cat(res_index,0)
    return res_score,res_index

