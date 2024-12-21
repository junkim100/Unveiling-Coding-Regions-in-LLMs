# Unveiling Coding Regions in Large Language Models

This repository contains a set of scripts and utilities designed to unveil linguistic regions in large language models. Below are step-by-step instructions for running the code successfully. Follow along to preprocess data, train models, select regions, and finally, assess the model's adaptability to damage.

## Project Structure

- **unveiling_code/**: Main project directory containing code organized by function.
  - **damage/**: Contains scripts related to model assessment.
  - **data_preprocess/**: Contains scripts for dataset preparation and tokenization.
  - **region_selection/**: Contains scripts for extracting linguistic regions.
  - **training/**: Contains scripts and utilities for model training and evaluation.

## Prerequisites

Ensure that you have Python and necessary libraries installed. You may require packages such as `transformers`, `datasets`, `torch`, etc. These can be installed via pip if not already available.

## Steps to Run the Code

Follow these steps in the specified order to achieve the complete result:

### Step 1: Data Preprocessing

1. **Navigate to the data preprocessing directory:**

   ```bash
   cd Unveiling-Coding-Regions-in-LLMs/data_preprocess
   ```

2. **Download Dataset for Processing:**

   Use the following command to download desired datasets:

   ```bash
   python create_code_dataset.py
   ```

3. **Preprocess Dataset:**

   Run the preprocess script to tokenize and prepare the dataset for training. For different languages, modify the script parameters accordingly:

   **Example for GO:**

   ```bash
   bash run_preprocess.sh tiny-codes go tokenizers/llama-3.2
   ```

   **Example for Java:**

   ```bash
   bash run_preprocess.sh tiny-codes java tokenizers/llama-3.2
   ```

### Step 2: Training

1. **Navigate to the training directory:**

   ```bash
   cd Unveiling-Coding-Regions-in-LLMs/training/further_training
   ```

2. **Run the Training Script:**

   Utilize the preprocessed dataset to calculate importance scores with the following command:

   ```bash
   bash code_train_core-10000.sh "tiny-codes" "llama-3.2" "meta-llama/Llama-3.2-3B-Instruct" "go"
   ```

   ```bash
   bash code_train_core-10000.sh "tiny-codes" "llama-3.2" "meta-llama/Llama-3.2-3B-Instruct" "java"
   ```

### Step 3: Region Selection

1. **Navigate to the region selection directory:**

   ```bash
   cd Unveiling-Coding-Regions-in-LLMs/region_selection
   ```

2. **Extract Core Linguistic Regions:**

   Execute these scripts to identify regions:

   ```bash
   python extract_accumulated_core_linguistic_region.py
   ```

   ```bash
   python extract_spot.py
   ```

### Step 4: Model Assessment

1. **Navigate to the damage directory:**

   ```bash
   cd Unveiling-Coding-Regions-in-LLMs/damage
   ```

2. **Run the Damage Assessment Script:**

   Use the following command to evaluate model robustness:

   ```bash
   python damage_model.py
   ```