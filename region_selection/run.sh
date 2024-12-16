#!/bin/bash

# Array of models and their corresponding paths
declare -A model_paths
model_paths["llama-3.1-8b"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_paths["codellama-7b"]="meta-llama/CodeLlama-7b-Instruct-hf"
model_paths["llama3.2-3b"]="meta-llama/Llama-3.2-3B-Instruct"
# model_paths["codellama-13b"]="meta-llama/CodeLlama-13b-Instruct-hf"

# Array of k values
k_values=(0.005 0.01 0.03 0.05)

# Array of programming languages
languages=("bash" "c#" "c++" "go" "java" "javascript" "julia" "ruby" "rust" "typescript")

# Function to join array elements with commas, excluding a specific element
join_array() {
    local exclude=$1
    shift
    local result=()
    for element in "$@"; do
        if [[ "$element" != "$exclude" ]]; then
            result+=("\"$element\"")
        fi
    done
    local IFS=,
    echo "'[${result[*]}]'"
}


# Loop through models
for model in "${!model_paths[@]}"; do
    model_path="${model_paths[$model]}"
    input_dir="/data_x/junkim100/code-spot/training/further_training/${model_path##*/}"

    # Loop through k values
    for k in "${k_values[@]}"; do
        # # Extract accumulated core linguistic region
        # echo Executing: python extract_accumulated_core_linguistic_region.py --model_name="$model" --original_model_path="$model_path" --k="$k" --input_dir="$input_dir"
        # python extract_accumulated_core_linguistic_region.py --model_name="$model" --original_model_path="$model_path" --k="$k" --input_dir="$input_dir"
        # echo "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"

        # # Extract spot for code
        # echo Executing: python extract_spot.py --model_name="$model" --original_model_path="$model_path" --core_path "/data_x/junkim100/code-spot/region_selection/code-region/$model/top$k" --instruct_path "/data_x/junkim100/code-spot/region_selection/${model_path##*/}" --k="$k" --input_dir="$input_dir" --code_or_lang "code"
        # python extract_spot.py --model_name="$model" --original_model_path="$model_path" --core_path "/data_x/junkim100/code-spot/region_selection/code-region/$model/top$k" --instruct_path "/data_x/junkim100/code-spot/region_selection/${model_path##*/}" --k="$k" --input_dir="$input_dir" --code_or_lang "code"
        # echo "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"

        # Loop through languages
        for lang in "${languages[@]}"; do
            # Create a comma-separated list of other languages
            other_langs=$(join_array "$lang" "${languages[@]}")

            # Extract accumulated monolingual region
            # echo Executing: python extract_accumulated_monolingual_region.py --model_name="$model" --original_model_path="$model_path" --k="$k" --input_dir="$input_dir" --language_base "$lang" --language_others \'"$other_langs"\'
            # python extract_accumulated_monolingual_region.py --model_name="$model" --original_model_path="$model_path" --k="$k" --input_dir="$input_dir" --language_base "$lang" --language_others $other_langs
            # echo "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"

            # Extract spot for language
            echo
            python extract_spot.py --model_name="$model" --original_model_path="$model_path" --core_path "/data_x/junkim100/code-spot/region_selection/lang-region/$model/$lang/top$k" --instruct_path "/data_x/junkim100/code-spot/region_selection/${model_path##*/}" --k="$k" --input_dir="$input_dir" --code_or_lang "$lang"
            echo "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"
        done
    done
done