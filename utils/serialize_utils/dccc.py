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
    data_dict: Dict[str, Dict[str, Any]],
) -> str:
    """
    Generate a single sentence describing one feature for a DCCC client.
    """
    feature_info = data_dict.get(feature_name, {})
    
    description = feature_info.get("description", feature_name)
    feature_type = feature_info.get("type", "")
    unit = feature_info.get("unit", "")
    
    # Handle missing values
    if pd.isna(value):
        return f"{description}. Value: missing."
    
    # Handle categorical, ordinal, and binary features
    if feature_type in ("categorical", "ordinal", "binary"):
        value_mapping = feature_info.get("values", {})
        
        numeric_value = format_numeric_value(value)
        meaning = value_mapping.get(numeric_value)
        
        if meaning:
            return f"{description}. Value: {numeric_value} ({meaning})."
        
        return f"{description}. Value: {numeric_value}."
    
    # Handle numeric and integer features
    numeric_value = format_numeric_value(value)
    
    if unit:
        return f"{description}. Value: {numeric_value} {unit}."
    
    return f"{description}. Value: {numeric_value}."


def generate_profile(
    row: pd.Series,
    data_dict: Dict[str, Dict[str, Any]],
) -> str:
    """
    Turn one DCCC record into a structured bullet list description.
    """
    sentences: List[str] = []
    
    for feature_name, value in row.items():
        feature_info = data_dict.get(feature_name, {})
        
        # Skip the client ID and target variable
        if feature_name == "ID":
            continue
        
        if feature_info.get("role") == "target":
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
    # Load DCCC metadata
    json_path = "datasets/dccc/raw/dccc_data_dictionary.json"
    xls_path = "datasets/dccc/raw/default of credit card clients.xls"
    
    data_dict = load_json(json_path)
    
    print("Data Dictionary sample:")
    
    for key, value in list(data_dict.items())[:5]:
        print(key, "->", value)
    
    # Load the DCCC dataset
    df = pd.read_excel(xls_path, header=1)
    
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
            "Default payment next month (label):",
            df.loc[i, "default payment next month"],
        )
        
        print("ApplicantProfile:")
        
        print(df.loc[i, "ApplicantProfile"])
