import os
import json
import pandas as pd
from dataclasses import asdict

def load_json(file_path: str) -> dict:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def save_json(file_path: str, data: list):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def update_json(file_path: str, key, value):
    # Read the data from the existing json file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    
    # Add new key-value pair
    data[key] = value
    
    # Write back into json file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def combine_jsons(input_file_path: str, output_file_path: str):
    combined_json = {}
    
    # Iterate through all files in the input_file_path
    for filename in os.listdir(input_file_path):
        if filename.endswith(".json"):
            file_path = os.path.join(input_file_path, filename)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    # If data is a list, update with each item
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                combined_json.update(item)
                            else:
                                print(f"Skipping non-dict item in {filename}")
                    # If data is a dict, update directly
                    elif isinstance(data, dict):
                        combined_json.update(data)
                    else:
                        print(f"Skipping non-dict data in {filename}")
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")
            except Exception as e:
                print(f"Unexpected error with {filename}: {e}")
    
    # Sort keys (numeric order for string keys)
    sorted_keys = sorted(combined_json.keys(), key=int)
    
    # Create new dict and populate in sorted order
    sorted_data = {}
    for key in sorted_keys:
        sorted_data[key] = combined_json[key]
    
    # Write sorted data to output_file_path
    try:
        with open(output_file_path, "w") as file:
            json.dump(sorted_data, file, indent=4, ensure_ascii=False)
        print(f"Successfully combined and sorted JSON files into {output_file_path}")
    except Exception as e:
        print(f"Error writing to {output_file_path}: {e}")

def load_jsonl(file_path: str) -> list:
    data = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line.strip()))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {file_path}: {e}")
    return data

def save_jsonl(file_path: str, data: list):
    with open(file_path, "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")

def update_jsonl(file_path: str, data: dict):
    with open(file_path, "a") as file:
        json.dump(data, file)
        file.write("\n")

def log_results(result: dict, pred_dict: dict, config: object, csv_result_dir: str, json_pred_dir: str, config_dir: str):
    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(csv_result_dir), exist_ok=True)
    os.makedirs(os.path.dirname(json_pred_dir), exist_ok=True)
    os.makedirs(os.path.dirname(config_dir), exist_ok=True)
    
    # Save result into csv
    result_df = pd.DataFrame([result])
    with open(csv_result_dir, mode="a") as f:
        result_df.to_csv(f, header=f.tell() == 0, index=False)
    
    # Save y_true, y_pred, and inference_time into json
    save_json(pred_dict, json_pred_dir)
    
    # Save config into json
    save_json(asdict(config), config_dir)
