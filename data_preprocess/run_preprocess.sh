# Must be ran inside the data_preprocess directory
# Example Usage: bash run_preprocess.sh "tiny-codes" "bash,java,python" "tokenizers/llama-3.1"

# Get the dataset name, languages, and tokenizer path from command-line arguments
DATASET_NAME=$1
LANGUAGE=$2
TOKENIZER_PATH=$3

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if dataset name is provided
if [ -z "$DATASET_NAME" ]; then
    echo "Please specify the dataset name as the first argument."
    exit 1
fi
if [ -z "$LANGUAGE" ]; then
    echo "Please specify the languages as the second argument."
    exit 1
fi
if [ -z "$TOKENIZER_PATH" ]; then
    echo "Please specify the tokenizer path as the third argument."
    exit 1
fi

DATASET_TYPE=(test train)

for type in "${DATASET_TYPE[@]}"
do
    for lang in "${LANGUAGE[@]}"
    do
        # Define the input file path based on dataset type
        INPUT_FILE_PATH="./dataset/${DATASET_NAME}/${type}/${lang}.jsonl"

        # Create output directory based on dataset type and language
        OUTPUT_DIR="./dataset/${DATASET_NAME}/preprocessed/${TOKENIZER_PATH##*/}/${type}/$lang/"
        mkdir -p "$OUTPUT_DIR"

        python preprocess-llama.py \
            --mode "write" \
            --file_path "$INPUT_FILE_PATH" \
            --save_prefix $type \
            --save_path "$OUTPUT_DIR" \
            --language $lang \
            --do_keep_newlines \
            --seq_length 2048 \
            --tokenizer_path "$TOKENIZER_PATH" \
            --num_workers 16
    done
done