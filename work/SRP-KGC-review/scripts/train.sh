
#!/bin/bash

# Configuration
TASK="FB15k237"
OUTPUT_DIR="./checkpoint/fb15k237_muti_token_8_1"
PRETRAINED_MODEL="./bert"  # Update this path to your BERT model location
DATA_DIR="./data/FB15k237"

# Training
echo "Starting training..."
python3 -u main_path_token_muti.py \
    --model-dir "${OUTPUT_DIR}" \
    --pretrained-model "${PRETRAINED_MODEL}" \
    --pooling mean \
    --lr 1e-5 \
    --use-link-graph \
    --train-path "${DATA_DIR}/train.txt.json" \
    --valid-path "${DATA_DIR}/test.txt.json" \
    --task ${TASK} \
    --batch-size 512 \
    --print-freq 200 \
    --additive-margin 0.02 \
    --use-amp \
    --finetune-t \
    --pre-batch 0 \
    --epochs 10 \
    --workers 4 \
    --max-to-keep 1 "$@"

# Evaluation
echo "Starting evaluation..."
neighbor_weight=0.0
rerank_n_hop=0
rerank_num=400

# List of model weights to evaluate
model_weights=("0.2" "0.4" "0.6" "0.8" "1")

for weight in "${model_weights[@]}"; do
    echo "Evaluating model with weight ${weight}..."
    model_path="./checkpoint/fb15k237_muti_token_8_${weight}/checkpoint_epoch9.mdl"
    
    if [ -f "${model_path}" ]; then
        python3 -u evaluate_path_token_muti.py \
            --task "${TASK}" \
            --is-test \
            --eval_model_path "${model_path}" \
            --neighbor-weight "${neighbor_weight}" \
            --rerank-n-hop "${rerank_n_hop}" \
            --train-path "${DATA_DIR}/train.txt.json" \
            --valid-path "${DATA_DIR}/test.txt.json" \
            --rerank-num "${rerank_num}" "$@"
    else
        echo "Model file not found: ${model_path}"
    fi
done

echo "Training and evaluation completed!"
