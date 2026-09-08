import os
import argparse
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_manager import DataManager
from datetime import datetime
from transformers.utils import logging
logging.set_verbosity_error()


def cal_Y_prob(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, generation_config, prompt_list):
    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in prompt_list
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    generated_output = model.generate(
        input_ids=inputs.input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_config
    )
    
    scores = generated_output.scores[0]
    probs = scores.softmax(dim=-1)
    
    Y_id = tokenizer.encode("Y", add_special_tokens=False)[0]
    N_id = tokenizer.encode("N", add_special_tokens=False)[0]
    
    Y_probs = [probs[i, Y_id].item() for i in range(probs.shape[0])]
    
    return Y_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset", help="Name of the dataset")
    parser.add_argument("--setting", type=str, choices=["inductive", "transductive"], default="inductive", help="Inductive or Transductive setting")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    parser.add_argument("--model_name", type=str, choices=["Qwen2-7B-Instruct", "Meta-Llama-3-8B-Instruct", "Qwen2-1.5B-Instruct"], default="Qwen2-7B-Instruct")
    parser.add_argument("--llm_type", type=str, choices=["sft", "base"], default="base")
    parser.add_argument("--subgraph_type", type=str, choices=["neighbor-only", "path-only", "combine"], default="combine")
    parser.add_argument("--path_type", type=str, choices=["degree", "no-degree"], default="degree")

    args = parser.parse_args()

    log_dir = f"logs_{args.model_name}_{args.llm_type}_{args.subgraph_type}_{args.path_type}"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_file = os.path.join(log_dir, f"log_{args.dataset}_{args.setting}_{args.train_size}_{timestamp}.txt")

    data_manager = DataManager(dataset=args.dataset, setting=args.setting, train_size=args.train_size, model_name=args.model_name, llm_type=args.llm_type)
    test_batches = data_manager.get_test_batches()

    model = AutoModelForCausalLM.from_pretrained(data_manager.model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(data_manager.model_path)
    generation_config = dict(
        temperature=0,
        top_k=0,
        top_p=0,
        do_sample=False,
        max_new_tokens=1,
    )

    llm_batch_size = 1
    sample_counter = 0

    def log_results(label, results, log):
        hit_at_1 = round(sum(1 for hits in results if hits == 1) / len(results), 3)
        mrr = round(sum(1 / hits for hits in results if hits != 0) / len(results), 3)
        log.write(f"{label} Hits results: {results}\n")
        log.write(f"{label} Hit@1: {hit_at_1}\n")
        log.write(f"{label} MRR: {mrr}\n")
    
    
    with open(log_file, 'w') as log:
        log.write(f"Using model: {data_manager.model_path}\n")
        
        hits_result_constraint = []
        hits_result_subgraph = []
        hits_result_average_ensemble = []
        TAR_infer_times = []
        SR_infer_times = []
        sample_counter = 0
        
        for idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
            constraint_prompts = [data_manager.build_constraint_prompt(test_triple) for test_triple in batch]
    
            if args.subgraph_type != "combine":
                raise ValueError("Only subgraph_type='combine' is supported in this simplified pipeline.")
            subgraph_prompts = [data_manager.build_subgraph_prompt(test_triple) for test_triple in batch]
    
            constraint_probs = []
            batch_infer_times = 0
            for i in range(0, len(constraint_prompts), llm_batch_size):
                batch_prompts = constraint_prompts[i:i + llm_batch_size]
                start_time = time.time()
                constraint_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))
                end_time = time.time()
                batch_infer_times += end_time - start_time
            TAR_infer_times.append(batch_infer_times)
    
            for prompt, prob in zip(constraint_prompts, constraint_probs):
                log.write(f"Sample {sample_counter} constraint Prompt: {prompt}\n")
                log.write(f"Sample {sample_counter} constraint 'Y' token Probability: {prob}\n")
                log.write("*" * 50 + "\n")
                sample_counter += 1
    
            sorted_constraint_indices = sorted(range(len(constraint_probs)), key=lambda i: constraint_probs[i], reverse=True)
            hits_position_constraint = sorted_constraint_indices.index(0) + 1 if 0 in sorted_constraint_indices else 0
            hits_result_constraint.append(hits_position_constraint)
    
            subgraph_probs = []
            batch_infer_times = 0
            for i in range(0, len(subgraph_prompts), llm_batch_size):
                batch_prompts = subgraph_prompts[i:i + llm_batch_size]
                start_time = time.time()
                subgraph_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))
                end_time = time.time()
                batch_infer_times += end_time - start_time
            SR_infer_times.append(batch_infer_times)
    
            for prompt, prob in zip(subgraph_prompts, subgraph_probs):
                log.write(f"Sample {sample_counter} Subgraph Prompt: {prompt}\n")
                log.write(f"Sample {sample_counter} Subgraph 'Y' token Probability: {prob}\n")
                log.write("*" * 50 + "\n")
                sample_counter += 1
    
            sorted_subgraph_indices = sorted(range(len(subgraph_probs)), key=lambda i: subgraph_probs[i], reverse=True)
            hits_position_subgraph = sorted_subgraph_indices.index(0) + 1 if 0 in sorted_subgraph_indices else 0
            hits_result_subgraph.append(hits_position_subgraph)
    

            combined_ranks = [sorted_constraint_indices.index(i) + sorted_subgraph_indices.index(i) for i in range(len(sorted_constraint_indices))]
            sorted_combined_indices = sorted(range(len(combined_ranks)), key=lambda i: combined_ranks[i])
            hits_position_average_ensemble = sorted_combined_indices.index(0) + 1 if 0 in sorted_combined_indices else 0
            hits_result_average_ensemble.append(hits_position_average_ensemble)
    
            log.write("*" * 50 + "\n")
            log.flush()
    
            if (idx + 1) % 100 == 0:
                log.write(f"\nMetrics after processing {idx + 1} batches:\n")
                log_results("constraint", hits_result_constraint, log)
                log_results("Subgraph", hits_result_subgraph, log)
                log_results("Average Ensemble", hits_result_average_ensemble, log)
                log.write("\n" + "=" * 50 + "\n")
                log.flush()
                # print to console
                print(f"\nMetrics after processing {idx + 1} batches:")
                print(f"constraint - Hit@1: {round(sum(1 for hits in hits_result_constraint if hits == 1) / len(hits_result_constraint),3)}, "
                      f"MRR: {round(sum(1 / hits for hits in hits_result_constraint if hits != 0)/len(hits_result_constraint),3)}")
                print(f"Subgraph - Hit@1: {round(sum(1 for hits in hits_result_subgraph if hits == 1) / len(hits_result_subgraph),3)}, "
                      f"MRR: {round(sum(1 / hits for hits in hits_result_subgraph if hits != 0)/len(hits_result_subgraph),3)}")
                print(f"Average Ensemble - Hit@1: {round(sum(1 for hits in hits_result_average_ensemble if hits == 1)/len(hits_result_average_ensemble),3)}, "
                      f"MRR: {round(sum(1 / hits for hits in hits_result_average_ensemble if hits != 0)/len(hits_result_average_ensemble),3)}")
    
        log.write("Final Results:\n")
        log.write("Proportion of constraint reasoning top 5: {}\n".format(sum(1 for hits in hits_result_constraint if hits <= 5) / len(hits_result_constraint)))
        log.write("Proportion of constraint reasoning top 10: {}\n".format(sum(1 for hits in hits_result_constraint if hits <= 10) / len(hits_result_constraint)))
        log_results("constraint", hits_result_constraint, log)
        log_results("Subgraph", hits_result_subgraph, log)
        log_results("Average Ensemble", hits_result_average_ensemble, log)
        log.write("Average time for constraint reasoning inference: {}\n".format(sum(TAR_infer_times) / len(TAR_infer_times)))
        log.write("Average time for subgraph reasoning inference: {}\n".format(sum(SR_infer_times) / len(SR_infer_times)))
        log.flush()


if __name__ == "__main__":
    main()