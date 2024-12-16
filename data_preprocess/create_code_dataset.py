import fire
from datasets import load_dataset
import json
from collections import defaultdict
from tqdm import tqdm
import os

class CodeSplitter:
    def __init__(self, dataset_name="nampdn-ai/tiny-codes", total_examples=1_630_000, train_ratio=0.8):
        self.dataset_name = dataset_name
        self.total_examples = total_examples
        self.train_ratio = train_ratio
        self.train_examples = int(total_examples * train_ratio)
        self.test_examples = total_examples - self.train_examples
        self.language_files = defaultdict(lambda: {"train": None, "test": None})
        self.id_counter = 0

        # Create directory for dataset files if it doesn't exist
        self.dataset_folder = self.dataset_name.split('/')[-1]

        # Run the code splitter with specified parameters.
        self.load_dataset()
        self.process_dataset()

    def load_dataset(self):
        self.ds = load_dataset(
            self.dataset_name, streaming=True, split="train", trust_remote_code=True
        )

    def process_dataset(self):
        progress_bar = tqdm(total=self.total_examples, unit="example")

        for example in self.ds:
            if self.dataset_name == "nampdn-ai/tiny-codes":
                language = example["programming_language"]
                try:
                    content = example["response"].split("```")[1]
                except:
                    content = example["response"]
            elif self.dataset_name == "grebniets123/codebase":
                language = example["path"].split("/")[-1].split(".")[-1]
                content = example["content"]
            # TODO: Implement other datasets
            else:
                raise ValueError("Invalid dataset")

            # Define file paths for train and test files
            os.makedirs(os.path.join("dataset", self.dataset_folder, "train"), exist_ok=True)
            os.makedirs(os.path.join("dataset", self.dataset_folder, "test"), exist_ok=True)
            train_file_path = os.path.join("dataset", self.dataset_folder, "train", f"{language.lower()}.jsonl")
            test_file_path = os.path.join("dataset", self.dataset_folder, "test", f"{language.lower()}.jsonl")

            # Open new files for train and test if not already open
            if self.language_files[language]["train"] is None:
                self.language_files[language]["train"] = open(train_file_path, "a")  # Append mode
            if self.language_files[language]["test"] is None:
                self.language_files[language]["test"] = open(test_file_path, "a")  # Append mode

            # Determine if the example should be in the train or test set
            file_type = "train" if self.id_counter < self.train_examples else "test"

            # Create the JSONL entry
            entry = {
                "id": self.id_counter,
                "language": language,
                "content": content,
            }

            # Write the entry to the appropriate file
            json.dump(entry, self.language_files[language][file_type])
            self.language_files[language][file_type].write("\n")

            self.id_counter += 1

            # Update the progress bar
            progress_bar.update(1)

            # Add description to progress bar every 10,000 examples
            if self.id_counter % 10_000 == 0:
                progress_bar.set_description(f"Processed {self.id_counter:,} examples")

            # Break the loop if we've processed all examples
            if self.id_counter >= self.total_examples:
                break

        # Close all open files
        for language, files in self.language_files.items():
            if files["train"]:
                files["train"].close()
            if files["test"]:
                files["test"].close()

        # Close the progress bar
        progress_bar.close()

        print("Finished processing the dataset")

if __name__ == '__main__':
    fire.Fire(CodeSplitter)