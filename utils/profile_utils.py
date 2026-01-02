import pandas as pd
from typing import Dict, Any, List


def describe_special_value(
    feature_name: str,
    numeric_value: int,
    special_values: Dict[int, str],
) -> str:
    """
    Describe a special value (-9, -8, -7) for a specific feature.
    
    We do NOT rely on the generic meta description.
    Instead, we craft feature-specific sentences and only use the special_values
    text as a reason (e.g. "no bureau record or no investigation").
    """
    # Get the meaning from special values
    meaning = special_values.get(numeric_value).strip()
    
    # Feature-specific phrasing
    if feature_name == "ExternalRiskEstimate":
        return f"The external risk estimate cannot be determined because: {meaning}."
    
    if feature_name == "MSinceOldestTradeOpen":
        return f"The number of months since the oldest trade was opened cannot be determined because: {meaning}."
    
    if feature_name == "MSinceMostRecentTradeOpen":
        return f"The number of months since the most recent trade was opened cannot be determined because: {meaning}."
    
    if feature_name == "AverageMInFile":
        return f"The average number of months that all of a applicant's credit accounts have been open cannot be determined because: {meaning}."
    
    if feature_name == "NumSatisfactoryTrades":
        return f"The number of satisfactory trades cannot be determined because: {meaning}."
    
    if feature_name == "NumTrades60Ever2DerogPubRec":
        return f"The number of trades 60+ ever to derogatory public record cannot be determined because: {meaning}."
    
    if feature_name == "NumTrades90Ever2DerogPubRec":
        return f"The number of trades 90+ ever to derogatory public record cannot be determined because: {meaning}."
    
    if feature_name == "PercentTradesNeverDelq":
        return f"The percentage of trades that have never been delinquent cannot be determined because: {meaning}."
    
    if feature_name == "MSinceMostRecentDelq":
        return f"The number of months since the most recent delinquency cannot be determined because: {meaning}."
    
    if feature_name == "MaxDelq2PublicRecLast12M":
        return f"The worst delinquency status over the past 12 months cannot be determined because: {meaning}."
    
    if feature_name == "MaxDelqEver":
        return f"The worst delinquency status ever recorded cannot be determined because: {meaning}."
    
    if feature_name == "NumTotalTrades":
        return f"The total number of trades (total number of credit accounts) cannot be determined because: {meaning}."
    
    if feature_name == "NumTradesOpeninLast12M":
        return f"The number of trades opened in the last 12 months cannot be determined because: {meaning}."
    
    if feature_name == "PercentInstallTrades":
        return f"The percentage of trades that are installment loans cannot be determined because: {meaning}."
    
    if feature_name == "MSinceMostRecentInqexcl7days":
        return f"The number of months since the most recent credit inquiry (excluding duplicates within 7 days) cannot be determined because: {meaning}."
    
    if feature_name == "NumInqLast6M":
        return f"The number of credit inquiries in the last 6 months cannot be determined because: {meaning}."
    
    if feature_name == "NumInqLast6Mexcl7days":
        return f"The number of credit inquiries in the last 6 months (excluding duplicates that are likely due to price comparison shopping within 7 days) cannot be determined because: {meaning}."
    
    if feature_name == "NetFractionRevolvingBurden":
        return f"The percentage of revolving credit utilization (revolving balance divided by credit limit) cannot be determined because: {meaning}."
    
    if feature_name == "NetFractionInstallBurden":
        return f"The percentage of installment credit utilization (installment balance divided by original loan amount) cannot be determined because: {meaning}."
    
    if feature_name == "NumRevolvingTradesWBalance":
        return f"The number of revolving trades with outstanding balance cannot be determined because: {meaning}."
    
    if feature_name == "NumInstallTradesWBalance":
        return f"The number of installment trades with outstanding balance cannot be determined because: {meaning}."
    
    if feature_name == "NumBank2NatlTradesWHighUtilization":
        return f"The number of bank or national trades with high utilization cannot be determined because: {meaning}."
    
    if feature_name == "PercentTradesWBalance":
        return f"The percentage of trades that currently carry a balance cannot be determined because: {meaning}."


def describe_feature(
    feature_name: str,
    value: Any,
    max_delq_dict: Dict[str, Dict[int, str]],
    special_values: Dict[int, str],
) -> str:
    """
    Generate a single sentence describing one feature for an applicant.
    Does NOT include a bullet prefix; that will be added when aggregating.
    """
    # Normalize numeric values
    numeric_value = int(value)
    
    # 1. Special values first (-9, -8, -7)
    if numeric_value in special_values:
        return describe_special_value(feature_name, numeric_value, special_values)
    
    # 2. MaxDelq mappings
    if feature_name in ("MaxDelq2PublicRecLast12M", "MaxDelqEver"):
        meaning = max_delq_dict[feature_name][numeric_value].lower()
        if feature_name == "MaxDelq2PublicRecLast12M":
            return f"The worst delinquency in the last 12 months is classified as '{meaning}'."
        else:
            return f"The worst delinquency ever recorded on this file is classified as '{meaning}'."
    
    # 3. Feature-specific templates (no meta description used)
    if feature_name == "ExternalRiskEstimate":
        return f"The external risk estimate is {numeric_value}, where lower scores indicate higher credit risk."
    
    if feature_name == "MSinceOldestTradeOpen":
        return f"The number of months since the oldest trade was opened is {numeric_value}."
    
    if feature_name == "MSinceMostRecentTradeOpen":
        return f"The number of months since the most recent trade was opened is {numeric_value}."
    
    if feature_name == "AverageMInFile":
        return f"The average number of months that all of a applicant's credit accounts have been open is {numeric_value}."
    
    if feature_name == "NumSatisfactoryTrades":
        return f"The number of satisfactory trades is {numeric_value}."
    
    if feature_name == "NumTrades60Ever2DerogPubRec":
        return f"The number of trades 60+ ever to derogatory public record is {numeric_value}."
    
    if feature_name == "NumTrades90Ever2DerogPubRec":
        return f"The number of trades 90+ ever to derogatory public record is {numeric_value}."
    
    if feature_name == "PercentTradesNeverDelq":
        return f"The percentage of trades that have never been delinquent is {numeric_value}%."
    
    if feature_name == "MSinceMostRecentDelq":
        return f"The number of months since the most recent delinquency is {numeric_value}."
    
    if feature_name == "NumTotalTrades":
        return f"The total number of trades (total number of credit accounts) is {numeric_value}."
    
    if feature_name == "NumTradesOpeninLast12M":
        return f"The number of trades opened in the last 12 months is {numeric_value}."
    
    if feature_name == "PercentInstallTrades":
        return f"The percentage of trades that are installment loans is {numeric_value}%."
    
    if feature_name == "MSinceMostRecentInqexcl7days":
        return f"The number of months since the most recent credit inquiry (excluding duplicates within 7 days) is {numeric_value}."
    
    if feature_name == "NumInqLast6M":
        return f"The number of credit inquiries in the last 6 months is {numeric_value}."
    
    if feature_name == "NumInqLast6Mexcl7days":
        return f"The number of credit inquiries in the last 6 months (excluding duplicates that are likely due to price comparison shopping within 7 days) is {numeric_value}."
    
    if feature_name == "NetFractionRevolvingBurden":
        return f"The percentage of revolving credit utilization (revolving balance divided by credit limit) is {numeric_value}%."
    
    if feature_name == "NetFractionInstallBurden":
        return f"The percentage of installment credit utilization (installment balance divided by original loan amount) is {numeric_value}%."
    
    if feature_name == "NumRevolvingTradesWBalance":
        return f"The number of revolving trades with outstanding balance is {numeric_value}."
    
    if feature_name == "NumInstallTradesWBalance":
        return f"The number of installment trades with outstanding balance is {numeric_value}."
    
    if feature_name == "NumBank2NatlTradesWHighUtilization":
        return f"The number of bank or national trades with high utilization is {numeric_value}."
    
    if feature_name == "PercentTradesWBalance":
        return f"The percentage of trades that currently carry a balance is {numeric_value}%."


def generate_profile(
    row: pd.Series,
    max_delq_dict: Dict[str, Dict[int, str]],
    special_values: Dict[int, str],
) -> str:
    """
    Turn one HELOC record (row) into a structured bullet list description.

    - Skips RiskPerformance (label).
    - Adds '- ' prefix only here.
    """
    sentences: List[str] = []
    
    for feature_name, value in row.items():
        if feature_name == "RiskPerformance":
            continue # ground truth label, skip
        
        sentence = describe_feature(
            feature_name,
            value,
            max_delq_dict=max_delq_dict,
            special_values=special_values,
        )
        
        if sentence:
            sentences.append(sentence)
        
    # bullets = "\n".join(f"- {s}" for s in sentences)
    # return "Applicant credit profile:\n" + bullets
    return "\n".join(f"- {s}" for s in sentences)


if __name__ == "__main__":
    # 1. Load HELOC metadata
    from utils.meta_utils import load_metadata
    excel_path = "dataset/heloc/raw/heloc_data_dictionary-2.xlsx"
    data_dict, max_delq_dict, special_vals = load_metadata(excel_path)
    
    # 2. Load your HELOC CSV
    csv_path = "dataset/heloc/raw/heloc_dataset_v1.csv" 
    df = pd.read_csv(csv_path)
    
    # 3. (Optional) check columns
    print("Columns:", df.columns.tolist())
    
    # 4. Create a new column with the natural language description for each row
    df["ApplicantProfile"] = df.apply(
        lambda row: generate_profile(
            row,
            max_delq_dict=max_delq_dict,
            special_values=special_vals,
        ),
        axis=1,
    )
    
    # 5. (Optional) inspect a few examples
    for i in range(3):
        print(f"\n=== Record {i} ===")
        print("RiskPerformance (label):", df.loc[i, "RiskPerformance"])
        print("ApplicantProfile:")
        print(df.loc[i, "ApplicantProfile"])
