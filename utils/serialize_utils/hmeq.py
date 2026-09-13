import pandas as pd
from typing import Dict, Any, List

from utils.json_utils import load_json


def format_numeric_value(value: Any) -> str:
    """
    Format a numeric value without unnecessary decimal places.
    """
    numeric_value = float(value)
    
    if numeric_value.is_integer():
        return str(int(numeric_value))
    
    return str(numeric_value)


def describe_feature(
    feature_name: str,
    value: Any,
    data_dict: Dict[str, str],
) -> str:
    """
    Generate a single sentence describing one feature for an HMEQ applicant.
    """
    description = data_dict.get(feature_name, feature_name)
    
    # Handle missing values
    if pd.isna(value) or value == -1:
        return f"{description}. Value: missing."
    
    # Handle categorical features
    if feature_name in ("REASON", "JOB"):
        return f"{description}. Value: {value}."
    
    # Handle numeric features
    numeric_value = format_numeric_value(value)
    
    return f"{description}. Value: {numeric_value}."


def generate_profile(
    row: pd.Series,
    data_dict: Dict[str, str],
) -> str:
    """
    Turn one HMEQ record into a structured bullet list description.
    """
    sentences: List[str] = []
    
    for feature_name, value in row.items():
        
        # Skip the target variable
        if feature_name == "BAD":
            continue
        
        sentence = describe_feature(
            feature_name,
            value,
            data_dict=data_dict,
        )
        
        if sentence:
            sentences.append(sentence)
    
    bullets = "\n".join(f"- {sentence}" for sentence in sentences)
    
    return bullets


if __name__ == "__main__":
    # Load HMEQ metadata
    json_path = "datasets/hmeq/raw/hmeq_data_dictionary.json"
    csv_path = "datasets/hmeq/raw/hmeq.csv"
    
    data_dict = load_json(json_path)
    
    print("Data Dictionary:")
    
    for key, value in data_dict.items():
        print(key, "->", value)
    
    # Load the HMEQ dataset
    df = pd.read_csv(csv_path)
    
    # Check columns
    print("Columns:", df.columns.tolist())
    
    # Create a new column with the natural language description for each row
    df["ApplicantProfile"] = df.apply(
        lambda row: generate_profile(
            row,
            data_dict=data_dict,
        ),
        axis=1,
    )
    
    # Inspect a few examples
    for i in range(3):
        print(f"\n=== Record {i} ===")
        
        print(
            "BAD (label):",
            df.loc[i, "BAD"],
        )
        
        print("ApplicantProfile:")
        
        print(df.loc[i, "ApplicantProfile"])
