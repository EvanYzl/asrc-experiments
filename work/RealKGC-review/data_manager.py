import os
import json
import random
import torch
import torch.nn as nn
import math
import numpy as np
from collections import defaultdict, deque
from sentence_transformers import SentenceTransformer
from prompt_templates import CONSTRIANT_REASON_PROMPT, SUBGRAPH_REASON_PROMPT

LLM_PATH = ""

class DataManager:
    def __init__(self, dataset="FB15k-237-subset", setting="inductive", train_size="full", model_name="Qwen2-7B-Instruct", llm_type="sft"):
        self.dataset = dataset
        self.model_name = model_name
        self.dataset_name = dataset.split("-")[0]
        self.dataset_path = f"/RealKGC-main/datasets/{dataset}" + ("-inductive" if setting=="inductive" else "")
        self.train_size = train_size
        self.model_path = f"{LLM_PATH}/{self.model_name}-{self.dataset_name}-{train_size}" if llm_type == "sft" else f"{LLM_PATH}/{self.model_name}"
        
        self.test_batch_size = 50
        self.max_type_triples = 3
        self.max_reason_paths = 6
        self.max_path_hops = 4
        self.each_count = 3

        self.entity2text = self._load_text_file("entity2text.txt")
        self.relation2text = self._load_text_file("relation2text.txt")

        self.train_set = self._load_triples(f"train_{self.train_size}.txt")
        self.path_set = self._load_triples("inductive_graph.txt") if setting=="inductive" else self.train_set
        self.valid_set = self._load_triples(f"valid.txt")
        self.test_set_head = self._load_triples(f"ranking_head.txt")
        self.test_set_tail = self._load_triples(f"ranking_tail.txt")
        self.test_set = self.test_set_head + self.test_set_tail
        self.triples = self.path_set

        self.relation2headtail_dict = self._load_relation2headtail_dict(self.path_set)
        self.entity2relationtail_dict = self._load_entity2relationtail_dict(self.path_set)
        self.relation_degree_dict = self._load_relation_degree_dict(self.path_set)
        self.close_path_file = f"paths/close_path.json" if setting=="inductive" else f"paths/close_path_train_size_{self.train_size}.json"
        self.close_path_dict = self._load_close_path_dict(self.close_path_file)

        self.embedding_model = SentenceTransformer(
            model_name_or_path='/RealKGC-main/bge-small-en-v1.5 ',
            device="cuda"
        )
        self.build_relation_index()
        self.build_triple_sentence_cache()
        self.build_neighbor_embedding_cache()
        self.precompute_test_embeddings()

    def build_relation_index(self):
        relations = sorted(list({r for (_, r, _) in self.triples}))
        self.rel2id = {r: idx for idx, r in enumerate(relations)}
        self.relation_dim = len(self.rel2id)

    def _load_text_file(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return dict(line.strip().split('\t', 1) for line in file if line.strip())

    def _load_triples(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip().split('\t') for line in file if line.strip()]

    def _load_relation2headtail_dict(self, triple_set):
        d = defaultdict(list)
        for h, r, t in triple_set:
            d[r].append([h, t])
        return d

    def _load_entity2relationtail_dict(self, triple_set):
        d = defaultdict(list)
        for h, r, t in triple_set:
            d[h].append((r, t, 1))
            d[t].append((r, h, -1))
        return d

    def _load_relation_degree_dict(self, triple_set):
        d = defaultdict(int)
        for _, r, _ in triple_set:
            d[r] += 1
        return d

    def _load_close_path_dict(self, filename):
        filepath = f"{self.dataset_path}/{filename}" 
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}


    def get_sentence_vec(self, sentence):
        if sentence in self.neighbor_sentence2vec:
            return self.neighbor_sentence2vec[sentence]
        vec = self.embedding_model.encode(
            [sentence],
            normalize_embeddings=True
        )[0]
        self.neighbor_sentence2vec[sentence] = vec
    
        return vec
        
    
    def triple_to_sentence(self, triple):
        head, relation, tail = triple
        head_text = self.entity2text.get(head, head)
        tail_text = self.entity2text.get(tail, tail)
        rel_text = self.relation2text.get(relation, relation)
    
        if self.dataset == "FB15k-237-subset":
            head_property = relation.split('/')[2] if '/' in relation else relation
            tail_property = relation.split('/')[-1] if '/' in relation else relation
            return f'("{tail_text}" is the {tail_property} of {head_property} "{head_text}")'    
        elif self.dataset in ["WN18RR-subset", "NELL-995-subset"]:
            return f"('{head_text}' {rel_text} '{tail_text}')"    
        else:
            return f"('{head_text}' {rel_text} '{tail_text}')"

    def build_triple_sentence_cache(self):
        self.triple2sentence = {}
        for h, r, t in self.triples:
            try:
                self.triple2sentence[(h, r, t)] = self.triple_to_sentence((h, r, t))
            except Exception:
                self.triple2sentence[(h, r, t)] = f'({h}, {r}, {t})'

    def build_neighbor_embedding_cache(self, batch_size=256):
        self.neighbor_sentence2vec = {}
        all_sentences = list(self.triple2sentence.values())
        if len(all_sentences) == 0:
            return
        vecs = self.embedding_model.encode(all_sentences, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
        for sent, vec in zip(all_sentences, vecs):
            self.neighbor_sentence2vec[sent] = vec

    def precompute_test_embeddings(self, batch_size=256):
        self.test_triple2vec = {}
        if not self.test_set:
            return
        sentences = [self.triple_to_sentence(tuple(tr)) for tr in self.test_set]
        vecs = self.embedding_model.encode(sentences, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
        for triple, vec in zip(self.test_set, vecs):
            self.test_triple2vec[tuple(triple)] = vec

    def generate_structure_constraint_sentence(self, head, tail):
    
        hop1 = self.entity2relationtail_dict.get(head, [])
        hop2 = []
        for r, n, direction in hop1:
            hop2.extend(self.entity2relationtail_dict.get(n, []))
        for r, n, direction in hop2:
            if n == tail:
                return (
                    f"Within the observed graph, a 2-hop structural path can be found between "
                    f"{self.entity2text.get(head, head)} and {self.entity2text.get(tail, tail)}. "
                    f"This provides limited structural support for the query triple, "
                    f"but does not constitute decisive evidence."
                )
        return (
            f"No explicit 2-hop structural path is observed between "
            f"{self.entity2text.get(head, head)} and {self.entity2text.get(tail, tail)} "
            f"in the current graph. This absence is treated as a weak negative structural signal, "
            f"rather than a hard contradiction."
        )

    
    def generate_background_constraint_evidence(self, test_triple, top_k=3, tau=0.3):
        h_q, r_q, t_q = test_triple
        v_eq = self.get_sentence_vec(self.entity2text.get(h_q, h_q))
        v_rq = self.get_sentence_vec(self.relation2text.get(r_q, r_q))
    
        candidates = []
        for r_b, n_i, direction in self.entity2relationtail_dict.get(h_q, []):
            if direction == 1:
                bg_triple = (h_q, r_b, n_i)
            else:
                bg_triple = (n_i, r_b, h_q)
            bg_triple_sent = self.triple_to_sentence(bg_triple)
            v_bi = self.get_sentence_vec(bg_triple_sent)
            

            n_text = self.entity2text.get(n_i, n_i)
            v_ni = self.get_sentence_vec(n_text)
            ent_sim = float(v_eq @ v_ni)
            if ent_sim < tau:
                continue
    
            rel_sim = float(v_rq @ v_bi)
            candidates.append((bg_triple_sent, ent_sim, rel_sim))
        
        if not candidates:
            return (
                f"No background triples highly relevant to the query entity "
                f"{self.entity2text.get(h_q, h_q)} were found. "
                f"Thus, no strong conflicting background evidence is observed."
            )    
        candidates.sort(key=lambda x: -x[2])
        top_neighbors = candidates[:top_k]
    
        text = (
            f"The following background triples are semantically related to the query entity "
            f"{self.entity2text.get(h_q, h_q)} and may potentially conflict with the query "
            f"relation {self.relation2text.get(r_q, r_q)}:\n"
        )
        for sent, ent_sim, rel_sim in top_neighbors:
            text += (
                f"- {sent} "
                f"(entity_sim={ent_sim:.3f}, relation_sim={rel_sim:.3f})\n"
            )    
        return text
    
    

    def build_CONSTRAINT_prompt(self, triple):
        fewshot_triples = self.same_relation_triple_finder(triple)
        fewshot_triples_sentence = '\n'.join(
            self.triple_to_sentence(ft) for ft in fewshot_triples
        )    
        background_evidence = self.generate_background_constraint_evidence(triple)
        structure_evidence = self.generate_2hop_structure_constraint_sentence(triple[0], triple[2])
        prompt = CONSTRAINT_REASON_PROMPT.replace("{fewshot_triples}", fewshot_triples_sentence)
        prompt = prompt.replace("{test_triple}", self.triple_to_sentence(triple))
        prompt = prompt.replace("{background_evidence}", background_evidence)
        prompt = prompt.replace("{structure_evidence}", structure_evidence)
        return prompt
        
    def bfs_paths(self, start, goal):    
        queue = deque([(start, [], 0, set([start]))])
        unique_paths = []
        seen = set()    
        while queue:
            current, path, hops, visited = queue.popleft()
            if hops >= self.max_path_hops:
                continue
            for relation, neighbor, direction in self.entity2relationtail_dict[current]:
                if direction == 1:
                    new_triple = (current, relation, neighbor)
                else:
                    new_triple = (neighbor, relation, current)    
                new_path = path + [new_triple]
                if neighbor == goal:
                    tuple_path = tuple(new_path)  
                    if tuple_path not in seen:
                        unique_paths.append(new_path)
                        seen.add(tuple_path)
                elif neighbor not in visited:
                    queue.append((
                        neighbor,
                        new_path,
                        hops + 1,
                        visited | {neighbor},
                    ))    
        return unique_paths

    def close_path_finder(self, triple):  
        head, relation, tail = triple
        head_tail = f"{head}-{tail}"
        close_paths = self.close_path_dict.get(head_tail, [])
        if not close_paths:
            return []
    
        v_rq = self.get_sentence_vec(self.relation2text.get(relation, relation))
        path_scores = []
    
        for path in close_paths:
            intermediate_entities = [n for _, _, n in path[1:-1]] if len(path) > 2 else []
            coupling_sims = []
            for e in intermediate_entities:
                v_e = self.get_sentence_vec(self.entity2text.get(e, e))
                cos_sim = float(v_e @ v_rq)                
                sigmoid_sim = 1 / (1 + math.exp(-cos_sim))
                log_sim = math.log(sigmoid_sim + 1e-8)
                coupling_sims.append(log_sim)
            
            if coupling_sims:
                coupling_score = sum(coupling_sims) / len(coupling_sims)
            else:
                coupling_score = 0.0 
    
            degree_sum = sum(self.relation_degree_dict.get(rel, 0) for _, rel, _ in path)
            path_scores.append((coupling_score, degree_sum, path))

        path_scores.sort(key=lambda x: (-x[0], x[1]))
        top_paths = [p for _, _, p in path_scores[:self.max_reason_paths]]
        return top_paths


    
    def linearize_triple(self, triple):
        return f"({self.entity2text[triple[0]]}, {self.relation2text[triple[1]]}, {self.entity2text[triple[2]]})"
        
    def build_subgraph_prompt(self, triple):
        neighbor_triples = self.neighbor_triple_finder(triple)
        close_paths = self.close_path_finder(triple)
        if not close_paths:
            return SUBGRAPH_REASON_PROMPT.format(
                neighbor_triples="\n".join(neighbor_triples),
                reasoning_paths="",
                test_triple=self.triple_to_sentence(triple)
            )
        
    def build_neighbor_prompt(self, triple):
        neighbor_triples = self.neighbor_triple_finder(triple)
        return NEIGHBOR_REASON_PROMPT.format(neighbor_triples="\n".join(neighbor_triples), test_triple=self.triple_to_sentence(triple))
    
    def build_close_path_prompt(self, triple):
        close_paths = self.close_path_finder(triple)
        reasoning_paths = "\n".join(
            " -> ".join(self.triple_to_sentence(triple) for triple in path)
            for path in close_paths
        )
        return CLOSE_PATH_REASON_PROMPT.format(reasoning_paths=reasoning_paths, test_triple=self.triple_to_sentence(triple))
    
    
    def build_vanilla_prompt(self, triple):
        return BASE_REASON_PROMPT.format(test_triple=self.triple_to_sentence(triple))
    
    def get_test_batches(self):
        return [self.test_set[i:i + self.test_batch_size] for i in range(0, len(self.test_set), self.test_batch_size)]
    
    def same_relation_triple_finder(self, test_triple):
        test_head, relation, test_tail = test_triple
        head_tail_pairs = self.relation2headtail_dict[relation]
        
        if len(head_tail_pairs) <= self.max_type_triples:
            return [[head, relation, tail] for head, tail in head_tail_pairs]
        
        used_heads = {test_head, test_tail}
        used_tails = {test_tail, test_head}
        used_pairs = set()
        selected_triples = []
        
        for head, tail in head_tail_pairs:
            if head not in used_heads and tail not in used_tails:
                selected_triples.append([head, relation, tail])
                used_heads.add(head)
                used_tails.add(tail)
                used_pairs.add((head, tail))
                if len(selected_triples) == self.max_type_triples:
                    return selected_triples
         
        for head, tail in head_tail_pairs:
            if (head, tail) not in used_pairs:
                if len(selected_triples) < self.max_type_triples:
                    selected_triples.append([head, relation, tail])
                    used_heads.add(head)
                    used_tails.add(tail)
                    used_pairs.add((head, tail))
                else:
                    break
        
        return selected_triples
        
    def neighbor_triple_finder(self, triple):
        triple = tuple(triple)
        head, relation, tail = triple
        head_name = self.entity2text.get(head, head)
        tail_name = self.entity2text.get(tail, tail)
        head_triples = self.entity2relationtail_dict.get(head, [])
        tail_triples = self.entity2relationtail_dict.get(tail, [])
        if triple not in self.triple2sentence:
            self.triple2sentence[triple] = self.triple_to_sentence(triple)
    
        triple_sentence = self.triple2sentence[triple]
        triple_vec = self.get_sentence_vec(triple_sentence)
    
        head_sentences = []
        for rel, nbr, direction in head_triples:
            if direction == 1:
                key = (head, rel, nbr)
            else:
                key = (nbr, rel, head)
    
            if key not in self.triple2sentence:
                self.triple2sentence[key] = self.triple_to_sentence(key)
    
            sent = self.triple2sentence[key]
    
            if head_name in sent:
                head_sentences.append(sent)
    
        tail_sentences = []
        for rel, nbr, direction in tail_triples:
            if direction == 1:
                key = (tail, rel, nbr)
            else:
                key = (nbr, rel, tail)
    
            if key not in self.triple2sentence:
                self.triple2sentence[key] = self.triple_to_sentence(key)
    
            sent = self.triple2sentence[key]
    
            if tail_name in sent:
                tail_sentences.append(sent)
    
        k = self.max_reason_paths // 2  
    
        def select_top_k(sentences, k):
            if len(sentences) <= k:
                return sentences
    
            vecs = np.vstack([self.get_sentence_vec(s) for s in sentences])
            sims = triple_vec @ vecs.T
            top_idx = np.argsort(-sims)[:k]
            return [sentences[i] for i in top_idx]
    
        top_head = select_top_k(head_sentences, k)
        top_tail = select_top_k(tail_sentences, k)
    
        return top_head + top_tail

    def neg_sampling(self, pos_triple, count):
        head, relation, tail = pos_triple
        
        entities = set()
        for triple in self.path_set:
            entities.add(triple[0])
            entities.add(triple[2])
        
        candidate_entities = entities - {head, tail}
        seen_triples = {tuple(triple) for triple in self.path_set}
        negative_samples = []
        
        for _ in range(count):
            while True:
                new_head = random.choice(list(candidate_entities))
                if (new_head, relation, tail) not in seen_triples:
                    seen_triples.add((new_head, relation, tail))
                    negative_samples.append((new_head, relation, tail))
                    break
        
        for _ in range(count):
            while True:
                new_tail = random.choice(list(candidate_entities))
                if (head, relation, new_tail) not in seen_triples:
                    seen_triples.add((head, relation, new_tail))
                    negative_samples.append((head, relation, new_tail))
                    break
                

        return negative_samples
