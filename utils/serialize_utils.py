import pandas as pd
from typing import Dict, Any, List


def describe_special_value(
    feature_name: str,
    numeric_value: int,
    special_values: Dict[int, str],
    data_dict: Dict[str, Dict[str, Any]],
) -> str:
    """
    Describe a special value (-9, -8, -7) for a specific feature.
    Now explicitly includes the numeric value to contextualize the monotonicity.
    """
    meaning = special_values.get(numeric_value, str(numeric_value)).strip().lower()
    desc = data_dict.get(feature_name, {}).get("description", feature_name)
    mono = data_dict.get(feature_name, {}).get("monotonicity", "")
    
    mono_tag = f" [Monotonicity: {mono}]" if mono else ""
    
    # Example: "ExternalRiskEstimate. Value: -9 (Reason: no bureau record). [Monotonicity: +1]"
    return f"{desc}. Value: {numeric_value} (Reason: {meaning}).{mono_tag}"


def describe_feature(
    feature_name: str,
    value: Any,
    max_delq_dict: Dict[str, Dict[int, str]],
    special_values: Dict[int, str],
    data_dict: Dict[str, Dict[str, Any]],
) -> str:
    """
    Generate a single sentence describing one feature for an applicant.
    Uses a 'Value: X' structure to prevent awkward grammar when descriptions contain full sentences.
    """
    numeric_value = int(value)
    
    # 1. Special values first
    if numeric_value in special_values:
        return describe_special_value(feature_name, numeric_value, special_values, data_dict)
    
    desc = data_dict.get(feature_name, {}).get("description", feature_name)
    mono = data_dict.get(feature_name, {}).get("monotonicity", "")
    mono_tag = f" [Monotonicity: {mono}]" if mono else ""
    
    # 2. MaxDelq mappings
    if feature_name in ("MaxDelq2PublicRecLast12M", "MaxDelqEver"):
        meaning = max_delq_dict.get(feature_name, {}).get(numeric_value, str(numeric_value)).lower()
        # Example: "Max Delq/Public Records Last 12 Months. Value: 7 (Classified as: 'current and never delinquent'). [Monotonicity: ...]"
        return f"{desc}. Value: {numeric_value} (Classified as: '{meaning}').{mono_tag}"
    
    # 3. Standard Features
    is_percentage = "percent" in feature_name.lower() or "fraction" in feature_name.lower()
    suffix = "%" if is_percentage else ""
    
    # Example: "Net Fraction Installment Burden. This is installment balance divided by original loan amount. Value: 66%. [Monotonicity: -1]"
    return f"{desc}. Value: {numeric_value}{suffix}.{mono_tag}"


def generate_profile(
    row: pd.Series,
    max_delq_dict: Dict[str, Dict[int, str]],
    special_values: Dict[int, str],
    data_dict: Dict[str, Dict[str, Any]],
) -> str:
    """
    Turn one HELOC record (row) into a structured bullet list description.
    """
    sentences: List[str] = []
    
    header = "(*Note: The [Monotonicity: X] tag indicates the constraint with respect to the probability of a Bad credit.*)\n"
    
    for feature_name, value in row.items():
        if feature_name == "RiskPerformance":
            continue 
        
        sentence = describe_feature(
            feature_name,
            value,
            max_delq_dict=max_delq_dict,
            special_values=special_values,
            data_dict=data_dict,
        )
        
        if sentence:
            sentences.append(sentence)
            
    bullets = "\n".join(f"- {s}" for s in sentences)
    return header + bullets


if __name__ == "__main__":
    # 1. Load HELOC metadata
    from utils.meta_utils import load_metadata
    excel_path = "datasets/heloc/raw/heloc_data_dictionary-2.xlsx"
    data_dict, max_delq_dict, special_vals = load_metadata(excel_path)
    
    # 2. Load your HELOC CSV
    csv_path = "datasets/heloc/raw/heloc_dataset_v1.csv" 
    df = pd.read_csv(csv_path)
    
    # 3. (Optional) check columns
    print("Columns:", df.columns.tolist())
    
    # 4. Create a new column with the natural language description for each row
    df["ApplicantProfile"] = df.apply(
        lambda row: generate_profile(
            row,
            max_delq_dict=max_delq_dict,
            special_values=special_vals,
            data_dict=data_dict, # <-- Pass data_dict down here
        ),
        axis=1,
    )
    
    # 5. (Optional) inspect a few examples
    for i in range(3):
        print(f"\n=== Record {i} ===")
        print("RiskPerformance (label):", df.loc[i, "RiskPerformance"])
        print("ApplicantProfile:")
        print(df.loc[i, "ApplicantProfile"])