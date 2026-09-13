import pandas as pd
from typing import Dict, Any, Tuple, List


def load_data_dictionary(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the 'Data Dictionary' sheet and return a dictionary of variables.
    
    Returns:
        {
          "VariableName": {
              "description": str,
              "monotonicity": str,
              "role": str
          },
          ...
        }
    """
    df = pd.read_excel(path, sheet_name="Data Dictionary")
    mono_col = "Monotonicity Constraint (with respect to probability of bad = 1)"
    
    # Drop completely empty rows and rows without a variable name
    df = df[df["Variable Names"].notna()]
    
    data_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        var_name = str(row["Variable Names"]).strip()
        if not var_name:
            continue
        
        description = ""
        if pd.notna(row["Description"]):
            description = str(row["Description"]).strip().strip('.')
            
            # Clean out "See tab..." instructions from the description
            if "See tab" in description:
                # Split at "See tab", take the first half, and strip trailing spaces/periods
                description = description.split("See tab")[0].strip(' .')
        
        monotonicity = ""
        if pd.notna(row[mono_col]):
            monotonicity = str(row[mono_col]).strip()
        
        role = ""
        if pd.notna(row["Role"]):
            role = str(row["Role"]).strip()
        
        data_dict[var_name] = {
            "description": description,
            "monotonicity": monotonicity,
            "role": role,
        }
    
    return data_dict


def parse_max_delq_block(df: pd.DataFrame, variable_name: str) -> Dict[int, str]:
    """
    Parse a single variable block from the 'Max Delq' sheet.
    
    The sheet has a structure like:
        row: variable_name
        row: "value   meaning ..."
        row+: code, meaning
    
    For example (conceptually):
        MaxDelq2PublicRecLast12M
        value   meaning
        0       derogatory comment
        1       120+ days delinquent
        5, 6    unknown delinquency
        ...
    
    Returns:
        { code_int: meaning_str, ... }
    """
    # Find the row index where the variable name appears
    idx_list = df.index[df.iloc[:, 0] == variable_name].tolist()
    if not idx_list:
        raise ValueError(f"Variable '{variable_name}' not found in 'Max Delq' sheet.")
    
    start_idx = idx_list[0] + 1
    
    # Skip header / empty rows until we reach rows with numeric codes
    while start_idx < len(df):
        val0 = df.iloc[start_idx, 0]
        val1 = df.iloc[start_idx, 1] if df.shape[1] > 1 else None
        
        text0 = "" if pd.isna(val0) else str(val0).strip().lower()
        
        # We want to skip:
        # - completely empty rows
        # - the header row that starts with "value"
        if pd.isna(val0) or text0.startswith("value"):
            start_idx += 1
            continue
        
        # We also require that there is a non-empty meaning in column 1
        if pd.isna(val1):
            start_idx += 1
            continue
        
        # If we reach here, this row should be a valid "code -> meaning" row
        break
    
    mapping: Dict[int, str] = {}
    row_idx = start_idx
    
    while row_idx < len(df):
        val0 = df.iloc[row_idx, 0]
        val1 = df.iloc[row_idx, 1] if df.shape[1] > 1 else None
        
        # Stop when codes or meaning are missing
        if pd.isna(val0) or pd.isna(val1):
            break
        
        codes_str = str(val0).strip()
        meaning = str(val1).strip()
        
        # Handle codes like "5, 6"
        for part in codes_str.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                code = int(part)
            except ValueError:
                # If the code is not an integer, skip it
                continue
            mapping[code] = meaning
            
        row_idx += 1
        
    return mapping


def load_max_delq_mappings(path: str) -> Dict[str, Dict[int, str]]:
    """
    Load the 'Max Delq' sheet and return mappings for:
      - MaxDelq2PublicRecLast12M
      - MaxDelqEver
    
    Returns:
        {
          "MaxDelq2PublicRecLast12M": { code_int: meaning_str, ... },
          "MaxDelqEver": { code_int: meaning_str, ... }
        }
    """
    df = pd.read_excel(path, sheet_name="Max Delq", header=None)
    
    mappings: Dict[str, Dict[int, str]] = {}
    mappings["MaxDelq2PublicRecLast12M"] = parse_max_delq_block(
        df, "MaxDelq2PublicRecLast12M"
    )
    mappings["MaxDelqEver"] = parse_max_delq_block(
        df, "MaxDelqEver"
    )
    
    return mappings


def load_special_values(path: str) -> Dict[int, str]:
    """
    Load the 'SpecialValues' sheet and return mapping of special codes to meaning.

    The sheet looks like:
        -9 No Bureau Record or No Investigation
        -8 No Usable/Valid Trades or Inquiries
        -7 Condition not Met (e.g. No Inquiries, No Delinquencies)
    
    Returns:
        {
          -9: "No Bureau Record or No Investigation",
          -8: "No Usable/Valid Trades or Inquiries",
          -7: "Condition not Met (e.g. No Inquiries, No Delinquencies)"
        }
    """
    df = pd.read_excel(path, sheet_name="SpecialValues", header=None)
    
    special_values: Dict[int, str] = {}
    for value in df.iloc[:, 0].dropna():
        text = str(value)
        # Split only on the first space: "code description..."
        first_space = text.find(" ")
        if first_space == -1:
            continue
        
        code_str = text[:first_space].strip()
        meaning = text[first_space + 1 :].strip()
        
        try:
            code = int(code_str)
        except ValueError:
            continue
        
        special_values[code] = meaning
    
    return special_values


def load_metadata(
    path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[int, str]], Dict[int, str]]:
    """
    Convenience helper to load all three metadata structures at once.
    
    Returns:
        data_dictionary, max_delq_mappings, special_values
    """
    data_dictionary = load_data_dictionary(path)
    max_delq_mappings = load_max_delq_mappings(path)
    special_values = load_special_values(path)
    return data_dictionary, max_delq_mappings, special_values


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
    excel_path = "datasets/heloc/raw/heloc_data_dictionary-2.xlsx"
    data_dict, max_delq_dict, special_vals = load_metadata(excel_path)
    print("Data Dictionary sample:")
    for k, v in list(data_dict.items())[:5]:
        print(k, "->", v)
    
    print("\nMax Delq mappings:")
    for var, mapping in max_delq_dict.items():
        for code, meaning in mapping.items():
            print(var, "->", {code: meaning})
    
    print("\nSpecial values mapping:")
    print(special_vals)
    
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
            data_dict=data_dict,
        ),
        axis=1,
    )
    
    # 5. (Optional) inspect a few examples
    for i in range(3):
        print(f"\n=== Record {i} ===")
        print("RiskPerformance (label):", df.loc[i, "RiskPerformance"])
        print("ApplicantProfile:")
        print(df.loc[i, "ApplicantProfile"])