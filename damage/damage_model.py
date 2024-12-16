# import torch
# import os
# import fire
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import setproctitle

# setproctitle.setproctitle("junkim100 damage_model")


# def process_file(weight_path, model, filename):
#     try:
#         weights = torch.load(weight_path)
#     except Exception as e:
#         print(f"Error loading weight file {weight_path}: {e}")
#         return

#     weights_np = weights.detach().float().cpu().numpy()
#     print(f"weights_np shape: {weights_np.shape}")
#     true_positions = np.where(weights_np == True)

#     print(f"true_positions: {true_positions}")
#     print(f"Type of true_positions: {type(true_positions)}")
#     print(f"Length of true_positions: {len(true_positions)}")
    
#     if len(true_positions) > 0:
#         print(f"Shape of true_positions[0]: {true_positions[0].shape}")
    
#     # Extract parameter name from filename
#     param_name = filename.replace(".pt", "")

#     # Find the corresponding parameter in the model and set to 0
#     for name, param in model.named_parameters():
#         if name == param_name:
#             if param.size() == weights.size():
#                 try:
#                     # Convert numpy indices to PyTorch tensor indices
#                     if len(true_positions) == 1:
#                         true_positions_tensor = torch.from_numpy(true_positions[0]).long()
#                         param.data.view(-1)[true_positions_tensor] = 0
#                     elif len(true_positions) == 2:
#                         true_positions_tensor = (torch.from_numpy(true_positions[0]).long(),
#                                                  torch.from_numpy(true_positions[1]).long())
#                         param.data[true_positions_tensor] = 0
#                     else:
#                         raise ValueError(f"Unexpected number of dimensions in true_positions: {len(true_positions)}")
                    
#                     print(f"Updated parameter: {name}")
#                     print(f"Number of values set to zero: {len(true_positions[0])}")
#                 except Exception as e:
#                     print(f"Error updating parameter: {e}")
#                     print(f"param.data shape: {param.data.shape}")
#             else:
#                 print(f"Size mismatch for {name}: expected {param.size()}, got {weights.size()}")
#             break
#     else:
#         print(f"No matching parameter found for {param_name}")




# def process_weight_files(
#     weights_folder: str = "/data_x/junkim100/projects/interpretability/Code-Spot/region_selection/llama-3.1/common_region/10000/top0.01",
#     original_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     output_dir: str = "/data_x/junkim100/projects/interpretability/Code-Spot/damage/llama-3.1-10000-0.01"
# ):
#     if not os.path.exists(weights_folder):
#         print(f"Error: Weights folder '{weights_folder}' not found.")
#         return

#     try:
#         model = AutoModelForCausalLM.from_pretrained(original_model)
#         tokenizer = AutoTokenizer.from_pretrained(original_model)
#     except Exception as e:
#         print(f"Error loading model or tokenizer '{original_model}': {e}")
#         return

#     for filename in os.listdir(weights_folder):
#         if filename.endswith(".pt"):
#             weight_path = os.path.join(weights_folder, filename)
#             process_file(weight_path, model, filename)
#             # break

#     print("Saving modified model and tokenizer...")
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print(f"Modified model and tokenizer saved to {output_dir}")

#     # try:
#     #     loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
#     #     loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
#     #     print("Successfully loaded the saved model and tokenizer for inference.")

#     #     test_input = "Hello, world!"
#     #     inputs = loaded_tokenizer(test_input, return_tensors="pt")
#     #     outputs = loaded_model.generate(**inputs, max_length=50)
#     #     print(
#     #         "Test inference output:",
#     #         loaded_tokenizer.decode(outputs[0], skip_special_tokens=True),
#     #     )
#     # except Exception as e:
#     #     print(f"Error verifying the saved model: {e}")


# if __name__ == "__main__":
#     fire.Fire(process_weight_files)




import torch
import os
import fire
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def process_file(weight_path, model, filename):
    try:
        weights = torch.load(weight_path)
        # print(f"weights: {weights}")
    except Exception as e:
        print(f"Error loading weight file {weight_path}: {e}")
        return

    weights_np = weights.detach().float().cpu().numpy()
    # print(f"weights_np: {weights_np}")
    # print(f"weights_np.shape: {weights_np.shape}")
    true_positions = np.where(weights_np > 0)
    # print(f"true_positions: {true_positions}")
    # print(f"size of true_positions: {len(true_positions[0])}")

    # Extract parameter name from filename
    param_name = filename.replace(".pt", "")

    # Find the corresponding parameter in the model and set to 0
    for name, param in model.named_parameters():
        if name == param_name:
            if param.size() == weights.size():
                param.data[true_positions] = 0
                # print(f"Updated parameter: {name}")
            else:
                print(
                    f"Size mismatch for {name}: expected {param.size()}, got {weights.size()}"
                )
            break
    else:
        print(f"No matching parameter found for {param_name}")


def process_weight_files(
    weights_folder: str, original_model: str, output_dir: str = "./damaged_models/"
):
    if not os.path.exists(weights_folder):
        print(f"Error: Weights folder '{weights_folder}' not found.")
        return

    try:
        model = AutoModelForCausalLM.from_pretrained(original_model)
        tokenizer = AutoTokenizer.from_pretrained(original_model)
    except Exception as e:
        print(f"Error loading model or tokenizer '{original_model}': {e}")
        return

    for filename in os.listdir(weights_folder):
        if filename.endswith(".pt"):
            weight_path = os.path.join(weights_folder, filename)
            process_file(weight_path, model, filename)
            # break

    print("Saving modified model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modified model and tokenizer saved to {output_dir}")

    # try:
    #     loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
    #     loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    #     print("Successfully loaded the saved model and tokenizer for inference.")

    #     test_input = "Hello, world!"
    #     inputs = loaded_tokenizer(test_input, return_tensors="pt")
    #     outputs = loaded_model.generate(**inputs, max_length=50)
    #     print(
    #         "Test inference output:",
    #         loaded_tokenizer.decode(outputs[0], skip_special_tokens=True),
    #     )
    # except Exception as e:
    #     print(f"Error verifying the saved model: {e}")


if __name__ == "__main__":
    fire.Fire(process_weight_files)
