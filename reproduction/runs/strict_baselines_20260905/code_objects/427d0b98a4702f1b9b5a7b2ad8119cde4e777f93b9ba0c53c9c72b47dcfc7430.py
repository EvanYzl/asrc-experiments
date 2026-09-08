import argparse
import numpy as np
import os
import pickle
import random
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW
from transformers import BertTokenizer

from importers.ea_ra_kgc import EaRaKgcData
from mBERT.Basic_Bert_Unit_model import Basic_Bert_Unit_model
from mBERT.Batch_TrainData_Generator import Batch_TrainData_Generator
from mBERT.Param import MbertParams


def ent2Tokens_gene(Tokenizer,ent_list,ent_name_max_length = MbertParams.DES_LIMIT_LENGTH - 2):
    ent2tokenids = dict()
    for ent_name in ent_list:
        token_ids = Tokenizer.encode(ent_name)[:ent_name_max_length]
        ent2tokenids[ent_name] = token_ids
    return ent2tokenids


def ent2bert_input(ent_ids,Tokenizer,ent2token_ids,des_max_length=MbertParams.DES_LIMIT_LENGTH):
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


def generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,
                                entid2data, index2entity,
                                nearest_sample_num = MbertParams.NEAREST_SAMPLE_NUM,
                                batch_size = MbertParams.CANDIDATE_GENERATOR_BATCH_SIZE):
    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        #langauge1 (KG1)
        train_emb1 = []
        for_candidate_emb1 = []
        for i in range(0,len(train_ent1s),batch_size):
            temp_emb = entlist2emb(Model,train_ent1s[i:i+batch_size],entid2data).cpu().tolist()
            train_emb1.extend(temp_emb)
        for i in range(0,len(for_candidate_ent2s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent2s[i:i+batch_size],entid2data).cpu().tolist()
            for_candidate_emb1.extend(temp_emb)

        #language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0,len(train_ent2s),batch_size):
            temp_emb = entlist2emb(Model,train_ent2s[i:i+batch_size],entid2data).cpu().tolist()
            train_emb2.extend(temp_emb)
        for i in range(0,len(for_candidate_ent1s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent1s[i:i+batch_size],entid2data).cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()

        #cos sim
        cos_sim_mat1 = cos_sim_mat_generate(train_emb1,for_candidate_emb1)
        cos_sim_mat2 = cos_sim_mat_generate(train_emb2,for_candidate_emb2)
        torch.cuda.empty_cache()
        #topk index
        _,topk_index_1 = batch_topk(cos_sim_mat1,topn=nearest_sample_num,largest=True)
        topk_index_1 = topk_index_1.tolist()
        _,topk_index_2 = batch_topk(cos_sim_mat2,topn=nearest_sample_num,largest=True)
        topk_index_2 = topk_index_2.tolist()
        #get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)
        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)

        #show
        # def rstr(string):
        #     return string.split(r'/resource/')[-1]
        # for e in train_ent1s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
        # for e in train_ent2s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
    print("get candidate using time: {:.3f}".format(time.time()-start_time))
    torch.cuda.empty_cache()
    return candidate_dict


def cos_sim_mat_generate(emb1,emb2,bs = 128):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2,dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2,dim=1)
    res_mat = batch_mat_mm(array_emb1,array_emb2.t(),bs=bs)
    return res_mat


def batch_mat_mm(mat1,mat2,bs=128):
    #be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat_1 = mat1[i:min(i+bs,axis_0)].cuda(MbertParams.CUDA_NUM)
        res = temp_div_mat_1.mm(mat2.cuda(MbertParams.CUDA_NUM))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat,0)
    return res_mat


def batch_topk(mat,bs=128,topn = 50,largest = False):
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


def hit_res(index_mat):
    ent1_num,ent2_num = index_mat.shape
    topk_n = [0 for _ in range(ent2_num)]
    for i in range(ent1_num):
        for j in range(ent2_num):
            if index_mat[i][j].item() == i:
                topk_n[j]+=1
                break
    for i in range(ent2_num):
        if i>0:
            topk_n[i]+=topk_n[i-1]
    topk_n = [round(x/ent1_num,5) for x in topk_n]
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_n[1 - 1],topk_n[10 - 1]),end="")
    if ent2_num >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_n[25 - 1]),end="")
    if ent2_num >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_n[50 - 1]),end="")
    print("")


def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s,pe2s,ne1s,ne2s in Train_gene:
        Optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model,pe1s,entid2data)
        pos_emb2 = entlist2emb(Model,pe2s,entid2data)
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
        del pos_emb1
        del pos_emb2

        neg_emb1 = entlist2emb(Model,ne1s,entid2data)
        neg_emb2 = entlist2emb(Model,ne2s,entid2data)
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
        del neg_emb1
        del neg_emb2

        label_y = -torch.ones(pos_score.shape).cuda(MbertParams.CUDA_NUM) #pos_score < neg_score
        batch_loss = Criterion( pos_score , neg_score , label_y )
        del pos_score
        del neg_score
        del label_y
        batch_loss.backward()
        Optimizer.step()

        all_loss += batch_loss.item() * batch_length
    all_using_time = time.time()-start_time
    return all_loss,all_using_time


def save(Model,train_ill,test_ill,entid2data,epoch_num, MODEL_SAVE_PATH):
    lnssrc = os.path.join(MODEL_SAVE_PATH, "model_epoch_" + str(epoch_num) + ".p")
    print("Model {} save in: ".format(epoch_num), lnssrc)
    Model.eval()
    torch.save(Model.state_dict(), lnssrc)
    lnsdst = os.path.join(MODEL_SAVE_PATH, "combined_mBERT.p")
    if os.path.islink(lnsdst):
        os.unlink(lnsdst);
    os.symlink(os.path.basename(lnssrc), lnsdst)
    other_data = [train_ill,test_ill,entid2data]
    pickle.dump(other_data,open(MODEL_SAVE_PATH + '/other_data.pkl',"wb"))
    print("Model {} save end.".format(epoch_num))


def test(Model,ent_ill,entid2data,batch_size,ent1,ent2,context = ""):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]
        diff_ent_1 = list(set(ent1) - set(ents_1))
        diff_ent_2 = list(set(ent2) - set(ents_2))
        ent1 = []
        ent1.extend(ents_1)
        ent1.extend(diff_ent_1)
        ent2 = []
        ent2.extend(ents_2)
        ent2.extend(diff_ent_2)

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        for_candidate_emb1 = []
        for i in range(0,len(ent2),batch_size):
            temp_emb = entlist2emb(Model,ent2[i:i+batch_size],entid2data).detach().cpu().tolist()
            for_candidate_emb1.extend(temp_emb)
            del temp_emb

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        for_candidate_emb2 = []
        for i in range(0,len(ent1),batch_size):
            temp_emb = entlist2emb(Model,ent1[i:i+batch_size],entid2data).detach().cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
            del temp_emb

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cos_sim_mat_generate(emb1,for_candidate_emb1,batch_size)
        score,top_index = batch_topk(res_mat,batch_size,topn = MbertParams.TOPK,largest=True)
        hit_res(top_index)

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cos_sim_mat_generate(emb2,for_candidate_emb2,batch_size)
        score,top_index = batch_topk(res_mat,batch_size,topn = MbertParams.TOPK,largest=True)
        hit_res(top_index)

    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")


def train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,entid2data,entities,
          model_save_path):
    print("start training...")
    for epoch in range(MbertParams.EPOCH_NUM):
        print("+++++++++++")
        print("Epoch: ",epoch)
        print("+++++++++++")
        pairs = list(train_ill.items())
        random.shuffle(pairs)
        train_ill=dict(pairs)
        for pair in train_ill:
            print(pair)
            #generate candidate_dict
            #(candidate_dict is used to generate negative example for train_ILL)
            train_ent1s = [e1 for e1,e2 in train_ill[pair]]
            train_ent2s = [e2 for e1,e2 in train_ill[pair]]
            for_candidate_ent1s = Train_gene[pair].ent_ids1
            for_candidate_ent2s = Train_gene[pair].ent_ids2
            print("train ent1s num: {} train ent2s num: {} for_Candidate_ent1s num: {} for_candidate_ent2s num: {}"
                  .format(len(train_ent1s),len(train_ent2s),len(for_candidate_ent1s),len(for_candidate_ent2s)))
            candidate_dict = generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,
                                                         for_candidate_ent2s,entid2data,Train_gene[pair].index2entity)
            Train_gene[pair].train_index_gene(candidate_dict) #generate training data with candidate_dict

            #train
            epoch_loss,epoch_train_time = ent_align_train(Model,Criterion,Optimizer,Train_gene[pair],entid2data)
            Optimizer.zero_grad()
            torch.cuda.empty_cache()
            print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch,epoch_loss,epoch_train_time))
        if epoch >= 0:
            if epoch !=0:
                save(Model,train_ill,test_ill,entid2data,epoch, model_save_path)
            for pair in test_ill:
                test(Model,test_ill[pair], entid2data, MbertParams.TEST_BATCH_SIZE,entities[pair[:2]],entities[pair[-2:]],context="EVAL IN TEST SET:"+pair)


def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(av):
    fixed(MbertParams.SEED_NUM)
    meta = EaRaKgcData(av.dbp5l_path)

    if not os.path.isdir(av.combined_bert_tuned_path):
        print("creating", av.combined_bert_tuned_path)
        os.mkdir(av.combined_bert_tuned_path, mode=0o755)

    VOCAB_PATH=os.path.join(av.bert_pretrained_path,
                            "bert-base-multilingual-cased-vocab.txt")
    Model = Basic_Bert_Unit_model(MbertParams.MODEL_INPUT_DIM,
                                  MbertParams.MODEL_OUTPUT_DIM,
                                  av.bert_pretrained_path)
    Model.cuda(MbertParams.CUDA_NUM)
    Criterion = nn.MarginRankingLoss(MbertParams.MARGIN,size_average=True)
    Optimizer = AdamW(Model.parameters(),lr=MbertParams.LEARNING_RATE)

    entities, entids = meta.get_entities()

    train_ill : dict[str, list[tuple[str]]] = dict()
    """key = LA-LB, val = listof (entA str, entB str)"""
    train_ea_dir = os.path.join(av.dbp5l_path, "seed_alignment_" +
                                str(av.ea_percent))
    for (langA, langB) in meta.lang_pairs:
        langAB = langA + "-" + langB
        train_ill[langAB] = list()
        with open(os.path.join(train_ea_dir, langAB + ".tsv")) as train_ea_file:
            for row2 in tqdm.tqdm(train_ea_file, desc="train_"+langAB):
                a=row2.split()
                x, y = int(float(a[0])), int(float(a[1]))
                train_ill[langAB].append((entities[langA][x],
                                          entities[langB][y]))

    # some redundancy still exists wrt previous loop
    test_ill : dict[str, list[tuple[str]]] = dict()
    """key = LA-LB, val = listof (entA str, entB str)"""
    test_ea_dir = os.path.join(av.dbp5l_path, "seed_alignment")
    for (langA, langB) in meta.lang_pairs:
        langAB = langA + "-" + langB
        test_ill[langAB] = list()
        with open(os.path.join(test_ea_dir, langAB + ".tsv")) as test_ea_file:
            for row2 in tqdm.tqdm(test_ea_file, desc="test_"+langAB):
                a=row2.split()
                x, y = int(float(a[0])), int(float(a[1]))
                test_ill[langAB].append((entities[langA][x],
                                         entities[langB][y]))

    # remove train from test
    for (langA, langB) in meta.lang_pairs:
        langAB = langA + "-" + langB
        test_ill[langAB]=list(set(test_ill[langAB]) - set(train_ill[langAB]))

    Train_gene = {}
    for pair in train_ill:
        train_ill[pair]=train_ill[pair]
        Train_gene[pair] = Batch_TrainData_Generator(train_ill[pair], entities[pair[:2]], entities[pair[-2:]],{},
        batch_size=MbertParams.TRAIN_BATCH_SIZE,neg_num=MbertParams.NEG_NUM)

    Tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH)
    ent2tokenids = ent2Tokens_gene(Tokenizer,entids)
    ent2data = ent2bert_input(entids,Tokenizer,ent2tokenids)

    train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,ent2data,entities,
          av.combined_bert_tuned_path)


def test_mbert_load(av):
    print("testing if model loads into GPU", av.combined_bert_tuned_path)
    tokenizer_path = "bert-base-multilingual-cased"
    bert_dir_path = av.bert_pretrained_path
    model_pkl_path = os.path.join(av.combined_bert_tuned_path, "combined_mBERT.p")
    Tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print("tokenizer loaded from", tokenizer_path)
    Model = Basic_Bert_Unit_model(768,300,bert_dir_path).cuda()
    print("model loaded from", bert_dir_path)
    Model.load_state_dict(torch.load(open(model_pkl_path,"rb")))
    print("state_dict loaded from", model_pkl_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_pretrained_path", required=True)
    ap.add_argument("--dbp5l_path", required=True)
    ap.add_argument("--ea_percent", type=int, required=False,
                    help="EA percent; if not provided, will sweep a range")
    ap.add_argument("--combined_base", required=True)
    av = ap.parse_args()

    if av.ea_percent:
        ea_percents = [av.ea_percent]
    else:
        ea_percents = [20, 40, 50, 60, 80]

    for ea_percent in ea_percents:
        av.ea_percent = ea_percent
        av.combined_bert_tuned_path = \
            os.path.join(av.combined_base,
                         ("bert_tuned_" + str(av.ea_percent) + "/"))
        main(av)
        test_mbert_load(av)
