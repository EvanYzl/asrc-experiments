'''
time CUDA_VISIBLE_DEVICES=1 python3 exp_get_pretrained_scores.py -d fb15k -m image_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}' -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt

x='-d fb15k -m image_model -a -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt';L=x.split(" ")
L = L[:5] + ['{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}']+L[5:]
import sys
sys.argv += L
##########NEW######
CUDA_VISIBLE_DEVICES=1 python analysis2.py -m CX -d yago
'''

import kb
import torch
import os
import csv
import numpy
import matplotlib.pyplot as plt
import argparse


from collections import defaultdict as dd

def check_type(type_e,e):
    if type_e == "UNK_<yagoLegalActor>":
        # test["UNK_<yagoLegalActor>"].add(e)
        return "people"
    if type_e == "UNK_<wordnet_movie_106613686>":
        # test["UNK_<wordnet_movie_106613686>"].add(e)
        return "film"
    if type_e == "person":
        # test["person"].add(e)
        return "people"
    if type_e == "UNK_<wordnet_location_100027167>":
        # test["UNK_<wordnet_location_100027167>"].add(e)
        return "location"

    if type_e == "UNK_<wordnet_city_108524735>":
        # test["UNK_<wordnet_city_108524735>"].add(e)
        return "location"

    if type_e == "UNK_<yagoGeoEntity>":
        # test["UNK_<yagoGeoEntity>"].add(e)
        return "location"

    return type_e

def get_entity_mid_name_type_rel_dict(model, dataset):
    mid_type=dd(str)
    mid_name=dd(str)
    name_mid=dd(str)
    dataset_root='./data/'+dataset+'/'
    with open("./data/"+dataset+"/entity_mid_name_type_typeid.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            mid_type[row[0]] = check_type(row[2])
            mid_name[row[0]] = row[1]
   
    print(model.keys())
    if "entity_map" in model.keys(): 
        entity_map = model['entity_map']
    else:
        print("May have issues here!!")
        entity_map = get_maps(dataset_root)

    if 'reverse_entity_map' in model:
        reverse_entity_map = model['reverse_entity_map']
    else:
        reverse_entity_map = {}
        for k,v in entity_map.items():
            reverse_entity_map[v] = k   
    ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map, rem=reverse_entity_map, add_unknowns=True)

    return mid_name,name_mid,mid_type, ktrain.reverse_relation_map, entity_map, reverse_entity_map
    

def get_maps(dataset_root):
    ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'))
    introduce_oov=1
    print("ALERT: introduce_oov", introduce_oov)

    if introduce_oov:
        if not "<OOV>" in ktrain.entity_map.keys():
            ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
            ktrain.nonoov_entity_count = ktrain.entity_map["<OOV>"]+1
    ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), ktrain.entity_map, ktrain.relation_map,
                  add_unknowns=not introduce_oov, nonoov_entity_count=ktrain.nonoov_entity_count)
    kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), ktrain.entity_map, ktrain.relation_map,
                   add_unknowns=not introduce_oov, nonoov_entity_count=ktrain.nonoov_entity_count)

    return ktrain.entity_map

def save_translated_file(file_name="", model=None,data=None, dataset=None):
    mid_name,name_mid,mid_type,reverse_relation_map,entity_map, reverse_entity_map = get_entity_mid_name_type_rel_dict(model, dataset)
    mid_name["<OOV>"] = "<OOV>"
    mid_type["<OOV>"] = "<OOV>"
    reverse_entity_map[len(reverse_entity_map)] = "<OOV>"   
    print(len(mid_name))
    print(len(mid_type))
    print(len(reverse_relation_map))
    print(len(entity_map))
    print(len(reverse_entity_map))
    all_data=dd(list)
    with open(file_name,"w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')    
        csv_writer.writerow(["r","e1","e2","e1_type","e2_type","e1P","e2P","e1P_type","e2P_type","e1Rank","e2Rank"])
        for r,e1,e2,e1p,e2p,e1r,e2r in data:
            csv_writer.writerow([reverse_relation_map[int(r)],mid_name[reverse_entity_map[int(e1)]],mid_name[reverse_entity_map[int(e2)]],mid_type[reverse_entity_map[int(e1)]],mid_type[reverse_entity_map[int(e2)]],mid_name[reverse_entity_map[int(e1p)]],mid_name[reverse_entity_map[int(e2p)]],mid_type[reverse_entity_map[int(e1p)]],mid_type[reverse_entity_map[int(e2p)]],e1r,e2r])
            all_data[(e1,r,e2)]=[reverse_relation_map[int(r)], mid_name[reverse_entity_map[int(e1)]],mid_name[reverse_entity_map[int(e2)]],mid_type[reverse_entity_map[int(e1)]],mid_type[reverse_entity_map[int(e2)]],mid_name[reverse_entity_map[int(e1p)]],mid_name[reverse_entity_map[int(e2p)]],mid_type[reverse_entity_map[int(e1p)]],mid_type[reverse_entity_map[int(e2p)]],e1r,e2r]
    return all_data

def save_both_file(model_data, file_name=""):
    with open(file_name,"w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["r","e1","e2","e1_type","e2_type","e1P_typedM","e2P_typedM","e1P_typeM","e2P_typeM","e1Rank_typedM","e2Rank_typedM","e1P_imageM","e2P_imageM","e1P_type_imageM","e2P_type_imageM","e1Rank_imageM","e2Rank_imageM"])
        for e1,r,e2 in model_data["typed_model"].keys():
            csv_writer.writerow(model_data["typed_model"][(e1,r,e2)]+model_data["image_model"][(e1,r,e2)][5:])


def type_analysis(model_data,verbose=1):
    type_performance = {"image_model":dd(int),"typed_model":dd(int),"all_type":dd(int)}
    for tup in model_data["typed_model"].keys():
        r,e1,e2,e1_type,e2_type,e1P_typedM,e2P_typedM,e1P_typeM,e2P_typeM,e1Rank_typedM,e2Rank_typedM,e1P_imageM,e2P_imageM,e1P_type_imageM,e2P_type_imageM,e1Rank_imageM,e2Rank_imageM = model_data["typed_model"][tup]+model_data["image_model"][tup][5:]
        type_performance["all_type"][e1_type]+=1
        type_performance["all_type"][e2_type]+=1
        type_performance["image_model"][e1_type]+=int(e1_type==e1P_type_imageM)
        type_performance["image_model"][e2_type]+=int(e2_type==e2P_type_imageM)
        type_performance["typed_model"][e1_type]+=int(e1_type==e1P_typeM)  
        type_performance["typed_model"][e2_type]+=int(e2_type==e2P_typeM)
        
    for types in type_performance["all_type"].keys():
        type_performance["image_model"][types] = round(100.0*type_performance["image_model"][types]/type_performance["all_type"][types],2)
        type_performance["typed_model"][types] = round(100.0*type_performance["typed_model"][types]/type_performance["all_type"][types],2)
    
    if verbose>0:
        f1 = open("./analysis_yago/improved_types.csv",'w')
        f2 = open("./analysis_yago/deter_types.csv",'w')
        csv_writer1 = csv.writer(f1,delimiter=',')
        csv_writer2 = csv.writer(f2,delimiter=',')
        improved = [];deter = []
        for types in type_performance["all_type"].keys():
            diff = round(type_performance["image_model"][types] - type_performance["typed_model"][types],2)
            if diff> 10:
                improved.append((types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]))
                csv_writer1.writerow([types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]])
            elif diff < -10:
                deter.append((types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]))
                csv_writer2.writerow([types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]])
        print(improved)
        print(deter)
        f1.close()
        f2.close()
    return type_performance

def rel_analysis(model_data,verbose=1):
    r_performance = {"image_model":dd(int),"typed_model":dd(int),"all":dd(int)}
    for tup in model_data["typed_model"].keys():
        r,e1,e2,e1_type,e2_type,e1P_typedM,e2P_typedM,e1P_typeM,e2P_typeM,e1Rank_typedM,e2Rank_typedM,e1P_imageM,e2P_imageM,e1P_type_imageM,e2P_type_imageM,e1Rank_imageM,e2Rank_imageM = model_data["typed_model"][tup]+model_data["image_model"][tup][5:]
        r_performance["all"][r]+=1
        s_t=int(e1_type==e1P_type_imageM)
        o_t=int(e2_type==e2P_type_imageM)
        r_performance["image_model"][r] += int(s_t and o_t)
        s_t=int(e1_type==e1P_typeM)
        o_t=int(e2_type==e2P_typeM)
        r_performance["typed_model"][r] += int(s_t and o_t)

    for r in r_performance["all"].keys():
        r_performance["image_model"][r] = round(100.0*r_performance["image_model"][r]/r_performance["all"][r],2)
        r_performance["typed_model"][r] = round(100.0*r_performance["typed_model"][r]/r_performance["all"][r],2)

    if verbose>0:
        f1 = open("./analysis_yago/improved_rels.csv",'w')
        f2 = open("./analysis_yago/deter_rels.csv",'w')
        csv_writer1 = csv.writer(f1,delimiter=',')
        csv_writer2 = csv.writer(f2,delimiter=',')
        improved = [];deter = []
        for r in r_performance["all"].keys():
            diff = round(r_performance["image_model"][r] - r_performance["typed_model"][r],2)
            if diff> 10:
                improved.append((r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]))
                csv_writer1.writerow([r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]])
            elif diff < -10:
                deter.append((r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]))
                csv_writer2.writerow([r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]])
        print("improved:")
        for ele in improved:
            print(ele)
        print("deter:")
        for ele in deter:
            print(ele)
        f1.close()
        f2.close()
    return r_performance

def ent_analysis(model_data,verbose=1):
    e_performance = {"image_model":dd(int),"typed_model":dd(int),"all":dd(int)}
    for tup in model_data["typed_model"].keys():
        r,e1,e2,e1_type,e2_type,e1P_typedM,e2P_typedM,e1P_typeM,e2P_typeM,e1Rank_typedM,e2Rank_typedM,e1P_imageM,e2P_imageM,e1P_type_imageM,e2P_type_imageM,e1Rank_imageM,e2Rank_imageM = model_data["typed_model"][tup]+model_data["image_model"][tup][5:]
        e1 = e1.replace(",",":")
        e2 = e2.replace(",",":")
        e_performance["all"][(e1,e1_type)]+=1
        e_performance["all"][(e2,e2_type)]+=1
        s_t=int(e1_type==e1P_type_imageM)
        o_t=int(e2_type==e2P_type_imageM)
        e_performance["image_model"][(e1,e1_type)] += s_t
        e_performance["image_model"][(e2,e2_type)] += o_t
        s_t=int(e1_type==e1P_typeM)
        o_t=int(e2_type==e2P_typeM)
        e_performance["typed_model"][(e1,e1_type)] += s_t
        e_performance["typed_model"][(e2,e2_type)] += o_t 

    for e in e_performance["all"].keys():
        e_performance["image_model"][e] = round(100.0*e_performance["image_model"][e]/e_performance["all"][e],2)
        e_performance["typed_model"][e] = round(100.0*e_performance["typed_model"][e]/e_performance["all"][e],2)

    if verbose>0:
        f1 = open("./analysis_yago/improved_ents.csv",'w')
        f2 = open("./analysis_yago/deter_ents.csv",'w')
        csv_writer1 = csv.writer(f1,delimiter=',')
        csv_writer2 = csv.writer(f2,delimiter=',')
        improved = [];deter = []
        for e in e_performance["all"].keys():
            diff = round(e_performance["image_model"][e] - e_performance["typed_model"][e],2)
            if diff> 10:
                improved.append((e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]))
                csv_writer1.writerow([e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]])
            elif diff < -10:
                deter.append((e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]))
                csv_writer2.writerow([e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]])
        print("improved:")
        for ele in improved:
            print(ele)
        print("**********************************************************************")
        print("deter:")
        for ele in deter:
            print(ele)
        f1.close()
        f2.close()
    return e_performance

def get_custom_dict_e():
    return {"freq":0, "out_degree":set(), "in_degree":set(), "e1_freq":0, "e2_freq":0}


def get_custom_dict_r():
    return {"freq":0, "out_degree":set(), "in_degree":set()}

def get_data_stats(model, dataset):
    _,name_mid,mid_type, _, _, _ = get_entity_mid_name_type_rel_dict(model, dataset)

    e_stats = dd(get_custom_dict_e)#{freq:0, out:set(), in:set(), e1_freq:0, e2_freq:0}
    type_stats = dd(get_custom_dict_e)
    r_stats = dd(get_custom_dict_r)#freq, out, in
    dataset_root='./data/'+dataset+'/'
    with open("./data/"+dataset+"/train.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            e1,r,e2 = row
            e_stats[e1]["freq"] += 1; e_stats[e1]["e1_freq"] += 1;
            e_stats[e1]["out_degree"].add(r);  
            e_stats[e2]["freq"] += 1; e_stats[e2]["e2_freq"] += 1;
            e_stats[e2]["in_degree"].add(r);    
            r_stats[r]["freq"] += 1
            r_stats[r]["in_degree"].add(e1); r_stats[r]["out_degree"].add(e2);             

            e1_type = mid_type[name_mid[e1]]; e2_type = mid_type[name_mid[e2]];
            type_stats[e1_type]["freq"] += 1; type_stats[e1_type]["e1_freq"] += 1;
            type_stats[e1_type]["out_degree"].add(r);
            type_stats[e2_type]["freq"] += 1; type_stats[e2_type]["e2_freq"] += 1;
            type_stats[e2_type]["in_degree"].add(r);

    for stats in [e_stats, r_stats, type_stats]: #CHECK!!
        for key in stats.keys():
            stats[key]["in_degree"] = len(stats[key]["in_degree"])
            stats[key]["out_degree"] = len(stats[key]["out_degree"])

    return e_stats, r_stats, type_stats

def get_list_dict():
    return dd(list)

def model_analysis(model_data, model, dataset, verbose=1):
    #e_stats, r_stats, type_stats =  get_data_stats(model, dataset)

    e_performance = dd(list); e1_performance = dd(list); e2_performance = dd(list)
    e_type_performance = dd(list); e1_type_performance = dd(list); e2_type_performance = dd(list)
    r_e1_performance = dd(list); r_e2_performance = dd(list); r_e_performance = dd(list);
    r_e1_sep_performance = dd(get_list_dict); r_e2_sep_performance = dd(get_list_dict); r_e_sep_performance = dd(get_list_dict);

    for tup in model_data.keys():
        r,e1,e2,e1_type,e2_type,e1P,e2P,e1P_type,e2P_type,e1Rank,e2Rank = model_data[tup]
        e_performance[e1].append(e1Rank); e1_performance[e1].append(e1Rank);
        e_performance[e2].append(e2Rank); e2_performance[e2].append(e2Rank);
    
        e_type_performance[e1_type].append(e1Rank); e1_type_performance[e1_type].append(e1Rank);
        e_type_performance[e2_type].append(e2Rank); e2_type_performance[e2_type].append(e2Rank);

        r_e1_performance[r].append(e1Rank); r_e_performance[r].append(e1Rank); 
        r_e2_performance[r].append(e2Rank); r_e_performance[r].append(e2Rank)
        
        r_e1_sep_performance[r][e1].append(e1Rank); r_e_sep_performance[r][e1].append(e1Rank); 
        r_e2_sep_performance[r][e2].append(e2Rank); r_e_sep_performance[r][e2].append(e2Rank)

    return e_performance, e1_performance, e2_performance, e_type_performance, e1_type_performance, e2_type_performance, r_e1_performance, r_e2_performance, r_e_performance, r_e1_sep_performance, r_e2_sep_performance, r_e_sep_performance 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Name of the dataset as in the analysis file name", required=True)
    parser.add_argument('-m', '--model', help="Name of the model as in the analysis file name", required=True)
    arguments = parser.parse_args()
    
    path = './analysis/'
    file_name="_analysis_r_e1e2_e1Predictede2Predicted_e1Ranke2Rank.csv"
    model_file="_best_valid_model.pt"
    models = []
    #models=["image_model","typed_model"]
    #models = ["CX_yago"]
    models.append(arguments.model+"_"+arguments.dataset)
    
    if not models:
        print("ERROR: No model to analyze")

    all_model_data = {}
    for model_name in models:
        saved_model = torch.load(path+model_name+model_file)
        with open(path+model_name+file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            tmp = save_translated_file(file_name=path+"readable_"+model_name+file_name,model=saved_model,data=csv_reader, dataset=arguments.dataset)
            all_model_data[model_name] = tmp
    #save_both_file(model_data, file_name=path+"readable_both_"+model_name+file_name)

    e_performance, e1_performance, e2_performance, e_type_performance, e1_type_performance, e2_type_performance, r_e1_performance, r_e2_performance, r_e_performance, r_e1_sep_performance, r_e2_sep_performance, r_e_sep_performance = model_analysis(all_model_data[arguments.model+"_"+arguments.dataset], saved_model, arguments.dataset, verbose=1) 

    e_stats, r_stats, type_stats =  get_data_stats(saved_model, arguments.dataset)

    x = [];y = []
    for ele in e_performance:
        y.append(numpy.mean(e_performance[ele]))
        x.append(int(e_stats[ele]["freq"]))
        
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)

    """
    print("SAVED COMBINED FILE")
    ta = type_analysis(all_model_data)
    print("TYPE ANALYSIS DONE")
    ra = rel_analysis(all_model_data)
    print("REL ANALYSIS DONE")
    ea = ent_analysis(all_model_data)
    print("ENT ANALYSIS DONE")
    """
