import pandas as pd
from typing import Dict, Any, List
 
 
def load_data_dictionary(
    excel_path: str,
) -> Dict[str, str]:
    """
    Load GMSC data dictionary from Excel.
    """
    dictionary_df = pd.read_excel(excel_path)
    
    data_dict = {}
    
    for _, row in dictionary_df.iterrows():
        data_dict[row["Variable Name"]] = row["Description"]
    
    return data_dict


def _format_numeric_value(value: Any) -> str:
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
    Generate a single sentence describing one feature for a GMSC applicant.
    """
    description = data_dict.get(
        feature_name,
        feature_name,
    )
    
    # Handle missing values
    if pd.isna(value):
        return f"{description}. Value: missing."
    
    # Handle numeric features
    numeric_value = _format_numeric_value(value)
    
    return f"{description}. Value: {numeric_value}."


def generate_profile(
    row: pd.Series,
    data_dict: Dict[str, str],
) -> str:
    """
    Turn one GMSC record into a structured bullet list description.
    """
    sentences: List[str] = []
    
    for feature_name, value in row.items():
        
        # Skip record ID
        if feature_name == "Unnamed: 0":
            continue
        
        # Skip target variable
        if feature_name == "SeriousDlqin2yrs":
            continue
        
        sentence = describe_feature(
            feature_name,
            value,
            data_dict=data_dict,
        )
        
        if sentence:
            sentences.append(sentence)
    
    bullets = "\n".join(
        f"- {sentence}"
        for sentence in sentences
    )
    
    return bullets


if __name__ == "__main__":
    
    # Load GMSC metadata
    excel_path = "datasets/gmsc/raw/Data Dictionary.xlsx"
    csv_path = "datasets/gmsc/raw/CreditScoring.csv"
    
    data_dict = load_data_dictionary(excel_path)
    
    print("Data Dictionary:")
    for key, value in data_dict.items():
        print(key, "->", value)
    
    # Load the GMSC dataset
    df = pd.read_csv(csv_path)
    
    # Check columns
    print("Columns:", df.columns.tolist())
    
    # Create natural language description for each row
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
        print("SeriousDlqin2yrs (label):", df.loc[i, "SeriousDlqin2yrs"])
        print("ApplicantProfile:", df.loc[i, "ApplicantProfile"])