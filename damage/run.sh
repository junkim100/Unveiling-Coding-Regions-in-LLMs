# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.1-8b/top0.005 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/code/top0.005
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.1-8b/top0.01 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/code/top0.01
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.1-8b/top0.03 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/code/top0.03
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.1-8b/top0.05 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/code/top0.05

# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-7b/top0.005 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/code/top0.005
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-7b/top0.01 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/code/top0.01
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-7b/top0.03 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/code/top0.03
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-7b/top0.05 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/code/top0.05`

# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.2-3b/top0.005 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/code/top0.005
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.2-3b/top0.01 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/code/top0.01
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.2-3b/top0.03 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/code/top0.03
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/llama-3.2-3b/top0.05 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/code/top0.05

# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-13b/top0.005 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/tcode/op0.005
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-13b/top0.01 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/code/top0.01
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-13b/top0.03 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/code/top0.03
# python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/code-spot/codellama-13b/top0.05 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/code/top0.05











# languages=("bash" "c#" "c++" "go" "java" "javascript" "julia" "ruby" "rust" "typescript")
languages=("go")

for lang in "${languages[@]}"; do
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.1-8b/${lang}/top0.005 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/${lang}/top0.005
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.1-8b/${lang}/top0.01 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/${lang}/top0.01
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.1-8b/${lang}/top0.03 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/${lang}/top0.03
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.1-8b/${lang}/top0.05 --original_model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.1-8b/${lang}/top0.05

    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-7b/${lang}/top0.005 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/${lang}/top0.005
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-7b/${lang}/top0.01 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/${lang}/top0.01
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-7b/${lang}/top0.03 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/${lang}/top0.03
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-7b/${lang}/top0.05 --original_model meta-llama/CodeLlama-7b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-7b/${lang}/top0.05

    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.2-3b/${lang}/top0.005 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/${lang}/top0.005
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.2-3b/${lang}/top0.01 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/${lang}/top0.01
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.2-3b/${lang}/top0.03 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/${lang}/top0.03
    python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/llama-3.2-3b/${lang}/top0.05 --original_model meta-llama/Llama-3.2-3B-Instruct --output_dir /data_x/junkim100/code-spot/damage/damaged_llama-3.2-3b/${lang}/top0.05

    # python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-13b/${lang}/top0.005 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/${lang}/top0.005
    # python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-13b/${lang}/top0.01 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/${lang}/top0.01
    # python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-13b/${lang}/top0.03 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/${lang}/top0.03
    # python damage_model.py --weights_folder /data_x/junkim100/code-spot/region_selection/lang-region/codellama-13b/${lang}/top0.05 --original_model meta-llama/CodeLlama-13b-Instruct-hf --output_dir /data_x/junkim100/code-spot/damage/damaged_codellama-13b/${lang}/top0.05
done