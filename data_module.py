
import os
import copy
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from utils.meta_utils import load_metadata
from utils.serialize_utils import generate_profile
from utils.json_utils import load_json, save_json


class HelocDataModule:
    def __init__(
        self,
    ):
        self.label_to_index = {"Good": 0, "Bad": 1}
        
        # Paths
        excel_path = "datasets/heloc/raw/heloc_data_dictionary-2.xlsx"
        csv_path = "datasets/heloc/raw/heloc_dataset_v1.csv"
        self.data_split_path = "datasets/heloc/data_splits/"
        os.makedirs(self.data_split_path, exist_ok=True)
        
        # Load HELOC metadata and dataset
        data_dict, max_delq_dict, special_vals = load_metadata(excel_path)
        self.df = pd.read_csv(csv_path)
        
        # Generate applicant profiles
        self.df["ApplicantProfile"] = self.df.apply(
            lambda row: generate_profile(
                row,
                data_dict=data_dict,
                max_delq_dict=max_delq_dict,
                special_values=special_vals,
            ),
            axis=1,
        )
        
        # Generate indices
        self.setup_splits()
    
    def setup_splits(self):
        self.indices = np.arange(len(self.df))
        if not os.path.exists(os.path.join(self.data_split_path, "valid_indices.json")):
            # Remove samples where all values are -9
            feature_cols = self.df.columns.drop(["RiskPerformance", "ApplicantProfile"])
            valid_mask = ~(self.df[feature_cols].eq(-9).all(axis=1))
            self.valid_indices = self.indices[valid_mask].tolist()
            
            # Split into training and test indices
            self.train_val_indices, self.test_indices = train_test_split(self.valid_indices, test_size=0.2, random_state=42)
            
            # Apply strict filtering to the train_val set: remove samples where any feature contains -7, -8, or -9
            train_val_df = self.df.loc[self.train_val_indices, feature_cols]
            strict_mask = ~(train_val_df.isin([-7, -8, -9]).any(axis=1))
            self.train_val_indices = train_val_df[strict_mask].index.tolist()
            
            # From the training set, create a validation set
            self.train_indices, self.val_indices = train_test_split(self.train_val_indices, test_size=0.2, random_state=42)
            
            # Save indices
            save_json(os.path.join(self.data_split_path, "valid_indices.json"), self.valid_indices)
            save_json(os.path.join(self.data_split_path, "train_indices.json"), self.train_indices)
            save_json(os.path.join(self.data_split_path, "val_indices.json"), self.val_indices)
            save_json(os.path.join(self.data_split_path, "train_val_indices.json"), self.train_val_indices)
            save_json(os.path.join(self.data_split_path, "test_indices.json"), self.test_indices)
        else:
            self.valid_indices = load_json(os.path.join(self.data_split_path, "valid_indices.json"))
            self.train_indices = load_json(os.path.join(self.data_split_path, "train_indices.json"))
            self.val_indices = load_json(os.path.join(self.data_split_path, "val_indices.json"))
            self.train_val_indices = load_json(os.path.join(self.data_split_path, "train_val_indices.json"))
            self.test_indices = load_json(os.path.join(self.data_split_path, "test_indices.json"))
    
    def get_feature_dataset(
        self,
        indices: List,
    ):
        X = self.df.iloc[indices].drop(columns=["RiskPerformance", "ApplicantProfile"]).values
        y = self.df.iloc[indices]["RiskPerformance"].map(self.label_to_index).values
        return X, y
    
    def get_profile_dataset(
        self,
        indices: list,
    ):
        """Prepares a Dataset object in the format of profiles and labels for the given indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
        
        Returns:
            dataset (Dataset): A Dataset object with the selected indices, numeric features, profiles, and labels.
        """
        dataset = {
            "indices": [],
            "numeric_features": [],
            "profiles": [],
            "labels": [],
        }
        for index in indices:
            dataset["indices"].append(str(index)) # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(self.df.iloc[index].drop(labels=["RiskPerformance", "ApplicantProfile"]).values.astype(int).tolist())
            dataset["profiles"].append(self.df.iloc[index]["ApplicantProfile"])
            dataset["labels"].append(self.label_to_index[self.df.iloc[index]["RiskPerformance"]])
        return Dataset.from_dict(dataset)
    
    def get_chat_dataset(
        self,
        indices: List,
        question_template: str=None,
        answer_template: str=None,
    ):
        """Prepares a Dataset object in the format of prompts and completions for the given indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
            question_template (str): The template for the question prompt. Defaults to None.
            answer_template (str): The template for the answer prompt. Defaults to None.
        
        Returns:
            dataset (Dataset): A Dataset object with the selected indices, numeric features, prompts, and completions.
        """
        dataset = {
            "indices": [],
            "numeric_features": [],
            "prompts": [],
            "completions": [],
        }
        for index in indices:
            dataset["indices"].append(str(index)) # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(self.df.iloc[index].drop(labels=["RiskPerformance", "ApplicantProfile"]).values.astype(int).tolist())
            dataset["prompts"].append(
                [
                    {
                        "content": question_template.format(profile=self.df.iloc[index]["ApplicantProfile"]),
                        "role": "user",
                    },
                ]
            )
            dataset["completions"].append(
                [
                    {
                        "content": f"{answer_template}{self.df.iloc[index]['RiskPerformance']}",
                        "role": "assistant",
                    },
                ]
            )
        return Dataset.from_dict(dataset)
    
    def get_dataloader(
        self,
        indices: List,
        tokenizer: AutoTokenizer,
        question_template: str,
        answer_template: str,
        batch_size: int,
    ):
        """Returns a DataLoader in Completion-Only and Prompt-Only Format based on the provided indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
            tokenizer (AutoTokenizer): The tokenizer to use.
            question_template (str): The template for the question prompt.
            answer_template (str): The template for the answer prompt.
            batch_size (int): The batch size for the DataLoader.
        
        Returns:
            dataloader: A DataLoader object with dynamic padding in Completion-Only and Prompt-Only Format.
        """
        def tokenize_fn(example, tokenizer, max_length=None):
            """Tokenizes a single example into input_ids, attention_mask, and labels.
            """
            prompt = example["prompts"]
            completion = example["completions"]
            
            # 1. Construct the "question only" message list -> to calculate prompt length
            prompt_ids = tokenizer.apply_chat_template(
                prompt, 
                tokenize=True, 
                add_generation_prompt=True,
                truncation=False,
            ).input_ids
            
            # 2. Construct the "full conversation" message list -> input_ids
            input_ids = tokenizer.apply_chat_template(
                prompt+completion, 
                tokenize=True,
                truncation=False,
            ).input_ids
            
            # 3. Generate labels and apply masking
            labels = copy.deepcopy(input_ids)
            prompt_len = len(prompt_ids)
            
            # Set the labels of the prompt tokens to -100 to ignore them in loss computation
            for i in range(len(labels)):
                if i < prompt_len:
                    labels[i] = -100
            
            # 4. Truncate if max_length is specified
            if not max_length is None:
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length] # no need to shift, when feed to hugging face CausalLM, it will automatically shift the labels by one to the right internally
            
            # 5. Create attention mask
            attention_mask = [1] * len(input_ids)
            
            # 6. Create prompt_ids for constrained decoding
            answer_ids = tokenizer(answer_template, add_special_tokens=False).input_ids
            prompt_ids = prompt_ids + answer_ids
            prompt_attention_mask = [1] * len(prompt_ids)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompts": prompt_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "numeric_features": example["numeric_features"],
                "indices": example["indices"],
            }
        
        def collate_fn(batch):
            """Custom collate function to handle mixed data types (indices, numeric features, and text)
            """
            # Extract non-standard fields to process them separately
            indices = [item.pop("indices") for item in batch]
            numeric_features = [item.pop("numeric_features") for item in batch]
            # Process prompts and prompt_attention_mask with DataCollatorWithPadding
            prompts = data_collator_pad([{"input_ids": item.pop("prompts"), "attention_mask": item.pop("prompt_attention_mask")} for item in batch])
            # Process input_ids, attention_mask, and labels with DataCollatorForLanguageModeling as labels needs to be padded with -100
            completions = data_collator_lm(batch)
            # Reassemble the batch
            batch = {}
            batch["input_ids"] = completions["input_ids"]
            batch["attention_mask"] = completions["attention_mask"]
            batch["labels"] = completions["labels"]
            batch["prompts"] = prompts["input_ids"]
            batch["prompt_attention_mask"] = prompts["attention_mask"]
            batch["numeric_features"] = torch.tensor(numeric_features, dtype=torch.bfloat16)
            batch["indices"] = indices
            return batch
        
        # Create a dataset
        dataset = self.get_chat_dataset(
            indices=indices, 
            question_template=question_template, 
            answer_template=answer_template,
        )
        dataset = dataset.map(
            tokenize_fn,
            fn_kwargs={"tokenizer": tokenizer, "max_length": None}, # pass addtional arguments to the tokenize_fn
            batched=False, # make tokenize_fn to process sample by sample due to complex tokenization logic
            remove_columns=["prompts", "completions"],
        )
        
        # DataCollatorForLanguageModeling uses right-padding only.
        # During SFT, both `input_ids` and `labels` are fed directly to `model.forward()` to compute the loss.
        # Since `model.forward()` does not automatically skip padding tokens when generating `position_ids`,
        # it simply assigns them sequentially based on the entire sequence length.
        # Therefore, right-padding is required to keep the `position_ids` of the valid text aligned across the batch.
        data_collator_lm = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            max_length=None, # set to None for dynamic padding to the longest sequence in the batch
            completion_only_loss=True,
            padding_free=False,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        
        # DataCollatorWithPadding pads the sequences based on the tokenizer settings.
        # ---------------------------------------------------
        # In `model.generate()`, padding tokens are automatically ignored when computing `position_ids`.
        # This prevents `position_ids` from being misaligned by leading pad tokens during generation.
        # Left-padding is strictly required for batched generation to ensure the causal LM always 
        # predicts the next token from the last valid (non-pad) token.
        # ---------------------------------------------------
        # If `model.forward()` is explicitly used to fetch next-token logits instead of `generate()`,
        # left-padding MUST still be used. With right-padding, `logits[:, -1, :]` would incorrectly
        # point to a padding token rather than the actual end of the text sequence.
        # ---------------------------------------------------
        # In our case, because we need to concat prefix and get next-token logits during inference.
        # For simplicity, we use a batch size of 1 during testing, though batching is also supported.
        data_collator_pad = DataCollatorWithPadding(
            tokenizer=tokenizer, 
            padding="longest"
        )
        
        # Generate DataLoader with the built dataset and collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, # no shuffle as train_test_split already shuffles
            collate_fn=collate_fn,
        )
        
        return dataloader
    
    def get_completion_dataloader(
        self,
        indices: List,
        tokenizer: AutoTokenizer,
        question_template: str,
        answer_template: str,
        batch_size: int,
    ):
        """Returns a DataLoader in Completion-Only Format based on the provided indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
            tokenizer (AutoTokenizer): The tokenizer to use.
            question_template (str): The template for the question prompt.
            answer_template (str): The template for the answer prompt.
            batch_size (int): The batch size for the DataLoader.
        
        Returns:
            dataloader: A DataLoader object with dynamic padding in Completion-Only Format.
        """
        def tokenize_fn(example, tokenizer, max_length=None):
            """Tokenizes a single example into input_ids, attention_mask, and labels.
            """
            prompt = example["prompts"]
            completion = example["completions"]
            
            # 1. Construct the "question only" message list -> to calculate prompt length
            prompt_ids = tokenizer.apply_chat_template(
                prompt, 
                tokenize=True, 
                add_generation_prompt=True,
                truncation=False,
            ).input_ids
            
            # 2. Construct the "full conversation" message list -> input_ids
            input_ids = tokenizer.apply_chat_template(
                prompt+completion, 
                tokenize=True,
                truncation=False,
            ).input_ids
            
            # 3. Generate labels and apply masking
            labels = copy.deepcopy(input_ids)
            prompt_len = len(prompt_ids)
            
            # Set the labels of the prompt tokens to -100 to ignore them in loss computation
            for i in range(len(labels)):
                if i < prompt_len:
                    labels[i] = -100
            
            # 4. Truncate if max_length is specified
            if not max_length is None:
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length] # no need to shift, when feed to hugging face CausalLM, it will automatically shift the labels by one to the right internally
            
            # 5. Create attention mask and position ids
            attention_mask = [1] * len(input_ids)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "numeric_features": example["numeric_features"],
                "indices": example["indices"],
            }
        
        def collate_fn(batch):
            """Custom collate function to handle mixed data types (indices, numeric features, and text)
            """
            # Extract non-standard fields to process them separately
            indices = [item.pop("indices") for item in batch]
            numeric_features = [item.pop("numeric_features") for item in batch]
            # Process input_ids, attention_mask, and labels with DataCollatorForLanguageModeling as labels needs to be padded with -100
            completions = data_collator(batch)
            # Reassemble the batch
            batch = {}
            batch["input_ids"] = completions["input_ids"]
            batch["attention_mask"] = completions["attention_mask"]
            batch["labels"] = completions["labels"]
            batch["indices"] = indices
            batch["numeric_features"] = torch.tensor(numeric_features, dtype=torch.bfloat16)
            return batch
        
        # Create a dataset
        dataset = self.get_chat_dataset(
            indices=indices, 
            question_template=question_template, 
            answer_template=answer_template,
        )
        dataset = dataset.map(
            tokenize_fn,
            fn_kwargs={"tokenizer": tokenizer, "max_length": None}, # pass addtional arguments to the tokenize_fn
            batched=False, # make tokenize_fn to process sample by sample due to complex tokenization logic
            remove_columns=["prompts", "completions"],
        )
        
        # DataCollatorForLanguageModeling uses right-padding only.
        # During SFT, both `input_ids` and `labels` are fed directly to `model.forward()` to compute the loss.
        # Since `model.forward()` does not automatically skip padding tokens when generating `position_ids`,
        # it simply assigns them sequentially based on the entire sequence length.
        # Therefore, right-padding is required to keep the `position_ids` of the valid text aligned across the batch.
        data_collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            max_length=None, # set to None for dynamic padding to the longest sequence in the batch
            completion_only_loss=True,
            padding_free=False,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        
        # Generate DataLoader with the built dataset and collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, # no shuffle as train_test_split already shuffles
            collate_fn=collate_fn,
        )
        
        return dataloader
    
    def get_prompt_dataloader(
        self,
        indices: List,
        tokenizer: AutoTokenizer,
        question_template: str,
        answer_template: str,
        batch_size: int,
    ):
        """Returns a DataLoader in Prompt-Only Format based on the provided indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
            tokenizer (AutoTokenizer): The tokenizer to use.
            question_template (str): The template for the question prompt.
            answer_template (str): The template for the answer prompt.
            batch_size (int): The batch size for the DataLoader.
        
        Returns:
            dataloader: A DataLoader object with dynamic padding in Prompt-Only Format.
        """
        def tokenize_fn(example, tokenizer, max_length=None):
            """Tokenizes a single example into input_ids, attention_mask, and labels.
            """
            prompt = example["prompts"]
            
            # Construct the "question only" message list -> to calculate prompt length
            prompt_ids = tokenizer.apply_chat_template(
                prompt, 
                tokenize=True, 
                add_generation_prompt=True,
                truncation=False,
            ).input_ids
            
            # Create prompt_ids for constrained decoding
            answer_ids = tokenizer(answer_template, add_special_tokens=False).input_ids
            prompt_ids = prompt_ids + answer_ids
            prompt_attention_mask = [1] * len(prompt_ids)
            
            return {
                "prompts": prompt_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "numeric_features": example["numeric_features"],
                "indices": example["indices"],
            }
        
        def collate_fn(batch):
            """Custom collate function to handle mixed data types (indices, numeric features, and text)
            """
            # Extract non-standard fields to process them separately
            indices = [item.pop("indices") for item in batch]
            numeric_features = [item.pop("numeric_features") for item in batch]
            # Process prompts and prompt_attention_mask with DataCollatorWithPadding
            prompts = data_collator([{"input_ids": item.pop("prompts"), "attention_mask": item.pop("prompt_attention_mask")} for item in batch])
            # Reassemble the batch
            batch = {}
            batch["prompts"] = prompts["input_ids"]
            batch["prompt_attention_mask"] = prompts["attention_mask"]
            batch["indices"] = indices
            batch["numeric_features"] = torch.tensor(numeric_features, dtype=torch.bfloat16)
            return batch
        
        # Create a dataset
        dataset = self.get_chat_dataset(
            indices=indices, 
            question_template=question_template, 
            answer_template=answer_template,
        )
        dataset = dataset.map(
            tokenize_fn,
            fn_kwargs={"tokenizer": tokenizer, "max_length": None}, # pass addtional arguments to the tokenize_fn
            batched=False, # make tokenize_fn to process sample by sample due to complex tokenization logic
            remove_columns=["prompts", "completions"],
        )
        
        # DataCollatorWithPadding pads the sequences based on the tokenizer settings.
        # ---------------------------------------------------
        # In `model.generate()`, padding tokens are automatically ignored when computing `position_ids`.
        # This prevents `position_ids` from being misaligned by leading pad tokens during generation.
        # Left-padding is strictly required for batched generation to ensure the causal LM always 
        # predicts the next token from the last valid (non-pad) token.
        # ---------------------------------------------------
        # If `model.forward()` is explicitly used to fetch next-token logits instead of `generate()`,
        # left-padding MUST still be used. With right-padding, `logits[:, -1, :]` would incorrectly
        # point to a padding token rather than the actual end of the text sequence.
        # ---------------------------------------------------
        # In our case, because we need to concat prefix and get next-token logits during inference.
        # For simplicity, we use a batch size of 1 during testing, though batching is also supported.
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, 
            padding="longest"
        )
        
        # Generate DataLoader with the built dataset and collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, # no shuffle as train_test_split already shuffles
            collate_fn=collate_fn,
        )
        
        return dataloader
