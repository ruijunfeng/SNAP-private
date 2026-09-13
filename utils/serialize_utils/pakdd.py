import pandas as pd
from typing import Dict, Any, List, Tuple


def load_variable_list(
    xls_path: str,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Load attribute names and descriptions directly from
    PAKDD2010_VariablesList.XLS.
    """
    variable_df = pd.read_excel(
        xls_path,
        engine="xlrd",
    )
    
    name_column = "Var_Title"
    description_column = "Var_Description"
    
    column_names = (
        variable_df[name_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    
    data_dict: Dict[str, str] = {}
    
    for _, row in variable_df.iterrows():
        if pd.isna(row[name_column]):
            continue
        
        feature_name = str(row[name_column]).strip()
        
        if pd.isna(row[description_column]):
            description = feature_name
        else:
            description = str(row[description_column]).strip(".")
        
        data_dict[feature_name] = description
    
    return column_names, data_dict


def normalize_value(value: Any) -> Any:
    """
    Normalize missing values in PAKDD.
    """
    if pd.isna(value):
        return pd.NA
    
    if isinstance(value, str):
        value = value.strip()
        
        if value == "" or value.upper() == "NULL":
            return pd.NA
    
    return value


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
    Generate a single sentence describing one feature.
    """
    description = data_dict.get(feature_name, feature_name)
    value = normalize_value(value)
    
    # Handle missing values
    if pd.isna(value):
        return f"{description}. Value: missing."
    
    # Handle numeric values
    try:
        formatted_value = format_numeric_value(value)
    except (TypeError, ValueError):
        formatted_value = str(value)
    
    return f"{description}. Value: {formatted_value}."


def generate_profile(
    row: pd.Series,
    data_dict: Dict[str, str],
) -> str:
    """
    Turn one PAKDD record into a structured bullet list.
    """
    sentences: List[str] = []
    
    for feature_name, value in row.items():
        
        # Skip identifier, non-informative field, and target
        if feature_name in (
            "ID_CLIENT",
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
            "TARGET_LABEL_BAD=1",
        ):
            continue
        
        sentence = describe_feature(
            feature_name,
            value,
            data_dict=data_dict,
        )
        
        sentences.append(sentence)
    
    return "\n".join(
        f"- {sentence}"
        for sentence in sentences
    )


def load_pakdd_dataframe(
    txt_path: str,
    column_names: List[str],
) -> pd.DataFrame:
    """
    Load the tab-separated, headerless PAKDD modeling dataset.
    """
    df = pd.read_csv(
        txt_path,
        sep="\t",
        header=None,
        dtype=object,
        na_values=["NULL"],
        keep_default_na=True,
        encoding="latin1",
    )
    
    if df.shape[1] != len(column_names):
        raise ValueError(
            f"Column number mismatch: "
            f"TXT has {df.shape[1]} columns, "
            f"but XLS defines {len(column_names)} attributes."
        )
    
    df.columns = column_names
    
    for column in df.columns:
        df[column] = df[column].map(normalize_value)
    
    return df


if __name__ == "__main__":
    xls_path = "datasets/pakdd/raw/PAKDD2010_VariablesList.XLS"
    txt_path = "datasets/pakdd/raw/PAKDD2010_Modeling_Data.txt"
    
    # Load attribute names and descriptions from the original XLS
    column_names, data_dict = load_variable_list(xls_path)
    
    # Load modeling data
    df = load_pakdd_dataframe(
        txt_path,
        column_names=column_names,
    )
    
    # Generate applicant profiles
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
        print("TARGET_LABEL_BAD=1:", df.loc[i, "TARGET_LABEL_BAD=1"])
        print("ApplicantProfile:")
        print(df.loc[i, "ApplicantProfile"])