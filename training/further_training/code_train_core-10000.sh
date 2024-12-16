# Must be ran inside the training/further_training directory
# Example Usage: bash code_train_core-10000.sh "tiny-codes" "llama-3.2" "meta-llama/Llama-3.2-3B-Instruct" "go"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
total_cards=8

DATASET_NAME=$1
TOKENIZER=$2
MODEL=$3
LANGUAGE=$4

if [ -z "$DATASET_NAME" ]; then
    echo "Please specify the dataset name as the first argument."
    exit 1
fi
if [ -z "$TOKENIZER" ]; then
    echo "Please specify the tokenizer as the second argument."
    exit 1
fi
if [ -z "$MODEL" ]; then
    echo "Please specify the model name as the third argument."
    exit 1
fi
if [ -z "$LANGUAGE" ]; then
    echo "Please specify the languages as the fourth argument."
    exit 1
fi

for lang in "${LANGUAGE[@]}"
do
    OUTPUT_DIR=${MODEL##*/}/$lang
    mkdir -p $OUTPUT_DIR
    echo $OUTPUT_DIR

    deepspeed accumulate_grad_mul_params-10000.py  \
        --model_name_or_path $MODEL \
        --pretrain_train_data_path /data_x/junkim100/projects/interpretability/Code-Spot/data_preprocess/dataset/${DATASET_NAME}/preprocessed/${TOKENIZER}/train/$lang/train \
        --pretrain_test_data_path /data_x/junkim100/projects/interpretability/Code-Spot/data_preprocess/dataset/${DATASET_NAME}/preprocessed/${TOKENIZER}/test/$lang/test \
        --max_seq_len 1024 \
        --learning_rate 5e-5 \
        --weight_decay 0.001 \
        --total_cards 8 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 5 \
        --per_device_eval_batch_size 2 \
        --zero_stage 2 \
        --seed 1234 \
        --deepspeed \
        --output_dir $OUTPUT_DIR \
        &> ${OUTPUT_DIR}/training_log.log
done
wait

