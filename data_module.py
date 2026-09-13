import os
import copy
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from utils.serialize_utils import heloc, hmeq, gmsc, pakdd
from utils.data_split_utils import generate_balance_splits
from utils.json_utils import load_json, save_json


class BaseDataModule:
    def get_numeric_dataloader(
        self,
        indices: List,
        batch_size: int,
        shuffle: bool=False,
    ):
        """Returns a DataLoader containing purely numeric features, labels, and indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle data. Defaults to False.
        
        Returns:
            dataloader: A DataLoader object yielding dict-format batches.
        """
        def collate_fn(batch):
            """Custom collate function to pack numeric features, labels, and indices.
            """
            indices = [item["indices"] for item in batch]
            numeric_features = torch.tensor([item["numeric_features"] for item in batch], dtype=torch.bfloat16)
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
            return {
                "indices": indices,
                "numeric_features": numeric_features,
                "labels": labels,
            }
        
        # Create a dataset by reusing get_profile_dataset
        dataset = self.get_profile_dataset(indices=indices)
        dataset = dataset.remove_columns(["profiles"])
        
        # Generate DataLoader with the built dataset and collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
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

    def get_dataloader(
        self,
        task_type: str,
        indices: List,
        batch_size: int,
        shuffle: bool = False,
        tokenizer: Optional[AutoTokenizer] = None,
        question_template: Optional[str] = None,
        answer_template: Optional[str] = None,
    ):
        """Returns a DataLoader based on the specified task type.
        
        Args:
            task_type (str): Type of task ('cls', 'completion', or 'prompt').
            indices (List): List of indices to select from the dataset.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle data. Defaults to False.
            tokenizer (AutoTokenizer, optional): The tokenizer to use.
            question_template (str, optional): The template for the question prompt.
            answer_template (str, optional): The template for the answer prompt.
        
        Returns:
            dataloader: A DataLoader object configured for the task.
        """
        # Dispatch to the corresponding dataloader builder based on task_type
        if task_type == "cls":
            return self.get_numeric_dataloader(
                indices=indices,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        elif task_type == "completion":
            return self.get_completion_dataloader(
                indices=indices,
                tokenizer=tokenizer,
                question_template=question_template,
                answer_template=answer_template,
                batch_size=batch_size,
            )
        elif task_type == "prompt":
            return self.get_prompt_dataloader(
                indices=indices,
                tokenizer=tokenizer,
                question_template=question_template,
                answer_template=answer_template,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")


class HELOCDataModule(BaseDataModule):
    def __init__(
        self,
    ):
        # Prepare dataset and directory
        self._prepare_dataset_and_directory()
        # Generate indices
        self._generate_splits()
    
    def _prepare_dataset_and_directory(self):
        # Define label mapping and number of features
        self.label_to_index = {"Good": 0, "Bad": 1}
        self.num_features = 23
        
        # Paths
        excel_path = "datasets/heloc/raw/heloc_data_dictionary-2.xlsx"
        csv_path = "datasets/heloc/raw/heloc_dataset_v1.csv"
        self.data_split_path = "datasets/heloc/data_splits/"
        os.makedirs(self.data_split_path, exist_ok=True)
        
        # Load HELOC metadata and dataset
        data_dict, max_delq_dict, special_vals = heloc.load_metadata(excel_path)
        self.df = pd.read_csv(csv_path)
        
        # Generate applicant profiles
        self.df["ApplicantProfile"] = self.df.apply(
            lambda row: heloc.generate_profile(
                row,
                data_dict=data_dict,
                max_delq_dict=max_delq_dict,
                special_values=special_vals,
            ),
            axis=1,
        )
    
    def _generate_splits(self):
        if not os.path.exists(os.path.join(self.data_split_path, "valid_indices.json")):
            # Remove samples where all values are -1
            feature_cols = self.df.columns.drop(["RiskPerformance", "ApplicantProfile"])
            valid_mask = ~(self.df[feature_cols].eq(-1).all(axis=1))
            self.valid_indices = np.arange(len(self.df))[valid_mask].tolist()
            
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


class PAKDDDataModule(BaseDataModule):
    def __init__(
        self,
    ):
        # Prepare dataset and directory
        self._prepare_dataset_and_directory()
        # Generate indices
        self._generate_splits()
    
    def _prepare_dataset_and_directory(self):
        # Define label mapping
        self.index_to_label = {0: "Good", 1: "Bad"}
        
        # Paths
        xls_path = "datasets/pakdd/raw/PAKDD2010_VariablesList.XLS"
        txt_path = "datasets/pakdd/raw/PAKDD2010_Modeling_Data.txt"
        self.data_split_path = "datasets/pakdd/data_splits/"
        os.makedirs(self.data_split_path, exist_ok=True)
        
        # Load PAKDD metadata and dataset
        column_names, data_dict = pakdd.load_variable_list(xls_path)
        self.df = pakdd.load_pakdd_dataframe(
            txt_path,
            column_names=column_names,
        )
        
        # Identify valid instances before filling missing values
        attribute_df = self.df.drop(
            columns=[
                "ID_CLIENT",
                "TARGET_LABEL_BAD=1",
                "CLERK_TYPE",
                "MARITAL_STATUS",
                "NACIONALITY",
                "POSTAL_ADDRESS_TYPE",
                "EDUCATION_LEVEL",
                "RESIDENCE_TYPE",
                "PROFESSION_CODE",
                "OCCUPATION_TYPE",
                "MATE_PROFESSION_CODE",
                "MATE_EDUCATION_LEVEL",
                "PRODUCT",
                "FLAG_OTHER_CARDS",
                "QUANT_SPECIAL_BANKING_ACCOUNTS",
            ]
        )
        self.valid_indices = attribute_df.index[
            ~attribute_df.isna().all(axis=1)
        ].tolist()
        
        # Generate applicant profiles before numerical encoding
        self.df["ApplicantProfile"] = self.df.apply(
            lambda row: pakdd.generate_profile(
                row,
                data_dict=data_dict,
            ),
            axis=1,
        )
        
        # Fill missing values with -1
        self.df = self.df.fillna(-1)
        
        # Prepare feature dataframe
        feature_df = self.df.drop(
            columns=[
                "ID_CLIENT",
                "TARGET_LABEL_BAD=1",
                "ApplicantProfile",
                "CLERK_TYPE",
                "MARITAL_STATUS",
                "NACIONALITY",
                "POSTAL_ADDRESS_TYPE",
                "EDUCATION_LEVEL",
                "RESIDENCE_TYPE",
                "PROFESSION_CODE",
                "OCCUPATION_TYPE",
                "MATE_PROFESSION_CODE",
                "MATE_EDUCATION_LEVEL",
                "PRODUCT",
                "FLAG_OTHER_CARDS",
                "QUANT_SPECIAL_BANKING_ACCOUNTS",
            ]
        ).copy()
        
        # Identify numeric and categorical features
        numeric_columns = []
        categorical_columns = []
        
        for column in feature_df.columns:
            converted = pd.to_numeric(
                feature_df[column],
                errors="coerce",
            )
            
            if converted.notna().all():
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)
        
        # Convert numeric features
        for column in numeric_columns:
            feature_df[column] = pd.to_numeric(
                feature_df[column],
                errors="coerce",
            )
        
        # Encode categorical features
        if categorical_columns:
            missing_mask = feature_df[categorical_columns] == -1
            
            feature_df[categorical_columns] = feature_df[
                categorical_columns
            ].astype(str)
            
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            
            feature_df[categorical_columns] = encoder.fit_transform(
                feature_df[categorical_columns]
            )
            
            feature_df[categorical_columns] = feature_df[
                categorical_columns
            ].mask(
                missing_mask,
                -1,
            )
        
        self.feature_df = feature_df.astype(float)
        self.num_features = len(self.feature_df.columns)
    
    def _generate_splits(self):
        if not os.path.exists(os.path.join(self.data_split_path, "valid_indices.json")):
            # Split valid instances into train_val and test sets
            self.train_val_indices, self.test_indices = train_test_split(
                self.valid_indices,
                test_size=0.2,
                random_state=42,
            )
            
            # Remove train_val instances with missing attribute values
            attribute_columns = [
                column
                for column in self.df.columns
                if column not in (
                    "ID_CLIENT",
                    "TARGET_LABEL_BAD=1",
                    "ApplicantProfile",
                    "CLERK_TYPE",
                    "MARITAL_STATUS",
                    "NACIONALITY",
                    "POSTAL_ADDRESS_TYPE",
                    "EDUCATION_LEVEL",
                    "RESIDENCE_TYPE",
                    "PROFESSION_CODE",
                    "OCCUPATION_TYPE",
                    "MATE_PROFESSION_CODE",
                    "MATE_EDUCATION_LEVEL",
                    "PRODUCT",
                    "FLAG_OTHER_CARDS",
                    "QUANT_SPECIAL_BANKING_ACCOUNTS",
                )
            ]
            
            self.train_val_indices = [
                index
                for index in self.train_val_indices
                if not (
                    self.df.iloc[index][attribute_columns] == -1
                ).any()
            ]
            
            # Split train_val into training and validation sets
            self.train_indices, self.val_indices = train_test_split(
                self.train_val_indices,
                test_size=0.2,
                random_state=42,
            )
            
            # Prepare labels for balancing
            dataset = [
                {"target": int(label)}
                for label in self.df["TARGET_LABEL_BAD=1"]
            ]
            
            # Balance training, validation, and test sets
            self.train_indices, self.val_indices, self.test_indices = generate_balance_splits(
                dataset,
                self.train_indices,
                self.val_indices,
                self.test_indices,
                label_key="target",
            )
            
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
        X = self.feature_df.iloc[indices].values
        y = self.df.iloc[indices]["TARGET_LABEL_BAD=1"].values.astype(int)
        
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
            dataset["indices"].append(str(index))  # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(self.feature_df.iloc[index].values.tolist())
            dataset["profiles"].append(self.df.iloc[index]["ApplicantProfile"])
            dataset["labels"].append(int(self.df.iloc[index]["TARGET_LABEL_BAD=1"]))
        
        return Dataset.from_dict(dataset)
    
    def get_chat_dataset(
        self,
        indices: List,
        question_template: str = None,
        answer_template: str = None,
    ):
        """Prepares a Dataset object in the format of prompts and completions for the given indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
        
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
            label = int(self.df.iloc[index]["TARGET_LABEL_BAD=1"])
            
            dataset["indices"].append(str(index))  # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(self.feature_df.iloc[index].values.tolist())
            
            dataset["prompts"].append(
                [
                    {
                        "content": question_template.format(
                            profile=self.df.iloc[index]["ApplicantProfile"]
                        ),
                        "role": "user",
                    },
                ]
            )
            
            dataset["completions"].append(
                [
                    {
                        "content": f"{answer_template}{self.index_to_label[label]}",
                        "role": "assistant",
                    },
                ]
            )
        
        return Dataset.from_dict(dataset)


class HMEQDataModule(BaseDataModule):
    def __init__(
        self,
    ):
        # Prepare dataset and directory
        self._prepare_dataset_and_directory()
        # Generate indices
        self._generate_splits()
    
    def _prepare_dataset_and_directory(self):
        # Define label mapping
        self.index_to_label = {0: "Good", 1: "Bad"}
        
        # Paths
        json_path = "datasets/hmeq/raw/hmeq_data_dictionary.json"
        csv_path = "datasets/hmeq/raw/hmeq.csv"
        self.data_split_path = "datasets/hmeq/data_splits/"
        os.makedirs(self.data_split_path, exist_ok=True)
        
        # Load HMEQ metadata and dataset
        data_dict = load_json(json_path)
        self.df = pd.read_csv(csv_path)
        
        # Fill missing values with -1
        self.df = self.df.fillna(-1)
        
        # Generate applicant profiles
        self.df["ApplicantProfile"] = self.df.apply(
            lambda row: hmeq.generate_profile(
                row,
                data_dict=data_dict,
            ),
            axis=1,
        )
        
        # Prepare numeric features
        feature_df = self.df.drop(columns=["BAD", "ApplicantProfile"])
        
        # Encode categorical features
        feature_df = pd.get_dummies(
            feature_df,
            columns=["REASON", "JOB"],
            dummy_na=True,
        )
        
        self.feature_df = feature_df.astype(float)
        self.num_features = len(self.feature_df.columns)
    
    def _generate_splits(self):        
        if not os.path.exists(os.path.join(self.data_split_path, "valid_indices.json")):
            # All HMEQ samples are valid
            self.valid_indices = np.arange(len(self.df)).tolist()
            
            # Split into training and test indices
            self.train_val_indices, self.test_indices = train_test_split(
                self.valid_indices,
                test_size=0.2,
                random_state=42,
            )
            
            # Remove instances with missing attribute values
            self.train_val_indices = [
                index
                for index in self.train_val_indices
                if not (self.df.iloc[index].drop(labels=["BAD", "ApplicantProfile"]) == -1).any()
            ]
            
            # From the training set, create a validation set
            self.train_indices, self.val_indices = train_test_split(
                self.train_val_indices,
                test_size=0.2,
                random_state=42,
            )
            
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
        X = self.feature_df.iloc[indices].values
        y = self.df.iloc[indices]["BAD"].values.astype(int)
        
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
            dataset["indices"].append(str(index))  # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(self.feature_df.iloc[index].values.tolist())
            dataset["profiles"].append(self.df.iloc[index]["ApplicantProfile"])
            dataset["labels"].append(int(self.df.iloc[index]["BAD"]))
        
        return Dataset.from_dict(dataset)
    
    def get_chat_dataset(
        self,
        indices: List,
        question_template: str = None,
        answer_template: str = None,
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
            label = int(self.df.iloc[index]["BAD"])
            
            dataset["indices"].append(str(index))  # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(self.feature_df.iloc[index].values.tolist())
            
            dataset["prompts"].append(
                [
                    {
                        "content": question_template.format(
                            profile=self.df.iloc[index]["ApplicantProfile"]
                        ),
                        "role": "user",
                    },
                ]
            )
            
            dataset["completions"].append(
                [
                    {
                        "content": f"{answer_template}{self.index_to_label[label]}",
                        "role": "assistant",
                    },
                ]
            )
        
        return Dataset.from_dict(dataset)


class GMSCDataModule(BaseDataModule):
    def __init__(
        self,
    ):
        # Prepare dataset and directory
        self._prepare_dataset_and_directory()
        # Generate indices
        self._generate_splits()
    
    def _prepare_dataset_and_directory(self):
        # Define label mapping and number of features
        self.index_to_label = {0: "Good", 1: "Bad"}
        self.num_features = 10
        
        # Paths
        excel_path = "datasets/gmsc/raw/Data Dictionary.xlsx"
        csv_path = "datasets/gmsc/raw/CreditScoring.csv"
        self.data_split_path = "datasets/gmsc/data_splits/"
        os.makedirs(self.data_split_path, exist_ok=True)
        
        # Load GMSC metadata and dataset
        data_dict = gmsc.load_data_dictionary(excel_path)
        self.df = pd.read_csv(csv_path)
        
        # Identify valid instances before filling missing values
        attribute_df = self.df.drop(
            columns=["Unnamed: 0", "SeriousDlqin2yrs"]
        )
        self.valid_indices = attribute_df.index[
            ~attribute_df.isna().all(axis=1)
        ].tolist()
        
        # Fill missing values with -1
        self.df = self.df.fillna(-1)
        
        # Generate applicant profiles
        self.df["ApplicantProfile"] = self.df.apply(
            lambda row: gmsc.generate_profile(
                row,
                data_dict=data_dict,
            ),
            axis=1,
        )
    
    def _generate_splits(self):
        if not os.path.exists(os.path.join(self.data_split_path, "valid_indices.json")):
            # Split valid instances into train_val and test sets
            self.train_val_indices, self.test_indices = train_test_split(
                self.valid_indices,
                test_size=0.2,
                random_state=42,
            )
            
            # Remove train_val instances with missing attribute values
            self.train_val_indices = [
                index
                for index in self.train_val_indices
                if not (
                    self.df.iloc[index]
                    .drop(labels=["Unnamed: 0", "SeriousDlqin2yrs", "ApplicantProfile"])
                    == -1
                ).any()
            ]
            
            # Split train_val into training and validation sets
            self.train_indices, self.val_indices = train_test_split(
                self.train_val_indices,
                test_size=0.2,
                random_state=42,
            )
            
            # Prepare labels for balancing
            dataset = [
                {"target": int(label)}
                for label in self.df["SeriousDlqin2yrs"]
            ]
            
            # Balance training, validation, and test sets
            self.train_indices, self.val_indices, self.test_indices = generate_balance_splits(
                dataset,
                self.train_indices,
                self.val_indices,
                self.test_indices,
                label_key="target",
            )
            
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
        feature_df = self.df.drop(
            columns=["Unnamed: 0", "SeriousDlqin2yrs", "ApplicantProfile"]
        )
        
        X = feature_df.iloc[indices].values.astype(float)
        y = self.df.iloc[indices]["SeriousDlqin2yrs"].values.astype(int)
        
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
        
        feature_df = self.df.drop(
            columns=["Unnamed: 0", "SeriousDlqin2yrs", "ApplicantProfile"]
        )
        
        for index in indices:
            dataset["indices"].append(str(index))  # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(
                feature_df.iloc[index].values.astype(float).tolist()
            )
            dataset["profiles"].append(self.df.iloc[index]["ApplicantProfile"])
            dataset["labels"].append(int(self.df.iloc[index]["SeriousDlqin2yrs"]))
        
        return Dataset.from_dict(dataset)
    
    def get_chat_dataset(
        self,
        indices: List,
        question_template: str = None,
        answer_template: str = None,
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
        
        feature_df = self.df.drop(
            columns=["Unnamed: 0", "SeriousDlqin2yrs", "ApplicantProfile"]
        )
        
        for index in indices:
            label = int(self.df.iloc[index]["SeriousDlqin2yrs"])
            
            dataset["indices"].append(str(index))  # used for result logging and loading, this needs to be a string.
            dataset["numeric_features"].append(
                feature_df.iloc[index].values.astype(float).tolist()
            )
            dataset["prompts"].append(
                [
                    {
                        "content": question_template.format(
                            profile=self.df.iloc[index]["ApplicantProfile"]
                        ),
                        "role": "user",
                    },
                ]
            )
            dataset["completions"].append(
                [
                    {
                        "content": f"{answer_template}{self.index_to_label[label]}",
                        "role": "assistant",
                    },
                ]
            )
        
        return Dataset.from_dict(dataset)
