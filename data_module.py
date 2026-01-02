
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
from utils.profile_utils import generate_profile
from utils.json_utils import load_json, save_json


class HelocDataModule:
    def __init__(
        self,
    ):
        self.label_map = {"Good": 0, "Bad": 1}
        
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
            valid_mask = ~(self.df[feature_cols].eq(-9).all(axis=1)) # | self.df[feature_cols].isin([-7, -8]).any(axis=1) # and samples contain -7 or -8
            self.valid_indices = self.indices[valid_mask].tolist()
            
            # Split into training and test indices
            self.train_val_indices, self.test_indices = train_test_split(self.valid_indices, test_size=0.2, random_state=42)
            
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
        y = self.df.iloc[indices]["RiskPerformance"].map(self.label_map).values
        return X, y
    
    def get_profile_dataset(
        self,
        indices: list,
    ):
        dataset = []
        for index in indices:
            dataset.append({
                "indices": index,
                "numeric_features": self.df.iloc[index].drop(labels=["RiskPerformance", "ApplicantProfile"]).values,
                "profiles": self.df.iloc[index]["ApplicantProfile"],
                "labels": self.label_map[self.df.iloc[index]["RiskPerformance"]],
            })
        return dataset
    
    def get_dataloader(
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
            prompt = example["prompt"]
            completion = example["completion"]
            
            # 1. Construct the "question only" message list -> to calculate prompt length
            prompt_ids = tokenizer.apply_chat_template(
                prompt, 
                tokenize=True, 
                add_generation_prompt=True,
                truncation=False,
            )
            
            # 2. Construct the "full conversation" message list -> input_ids
            input_ids = tokenizer.apply_chat_template(
                prompt+completion, 
                tokenize=True,
                truncation=False,
            )
            
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
                    labels = labels[:max_length]
            
            # 5. Create attention mask
            attention_mask = [1] * len(input_ids)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "numeric_features": example["numeric_features"],
                "indices": example["indices"],
            }
        
        def collate_fn(features):
            """Separate indices and numeric features from data collator
            """
            indices = [f["indices"] for f in features]
            numeric_features = [f["numeric_features"] for f in features]
            # Remove from dict so collator doesn't complain
            for f in features:
                f.pop("indices")
                f.pop("numeric_features")
            # Use the data collator to get input_ids and labels
            batch = data_collator(features)
            # Add back indices and numeric features
            batch["indices"] = indices
            batch["numeric_features"] = torch.tensor(numeric_features, dtype=torch.float32)
            return batch
        
        # create a dataset
        numeric_features = []
        prompts = []
        completions = []
        for index in indices:
            numeric_features.append(self.df.iloc[index].drop(labels=["RiskPerformance", "ApplicantProfile"]).values.astype(int).tolist())
            prompts.append(
                [
                    {
                        "content": question_template.format(profile=self.df.iloc[index]["ApplicantProfile"]),
                        "role": "user",
                    },
                ]
            )
            completions.append(
                [
                    {
                        "content": f"{answer_template}{self.df.iloc[index]['RiskPerformance']}",
                        "role": "assistant",
                    },
                ]
            )
        dataset = Dataset.from_dict({"indices": indices, "numeric_features": numeric_features, "prompt": prompts, "completion": completions})
        dataset = dataset.map(
            tokenize_fn,
            fn_kwargs={"tokenizer": tokenizer, "max_length": None},
            batched=False, # make tokenize_fn to process sample by sample
            remove_columns=["prompt", "completion"],
        )
        
        # create a collator for dynamic padding 
        data_collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=True,
            padding_free=False,
        )
        
        # create a dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, # no shuffle as train_test_split already shuffles
            collate_fn=collate_fn,
        )
        
        return dataloader
