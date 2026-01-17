from dataclasses import dataclass, field

from methods.base.config import BaseConfig

examples = """
Here is a example of bad credit user profile:
- The external risk estimate is 66, where lower scores indicate higher credit risk.
- The number of months since the oldest trade was opened is 278.
- The number of months since the most recent trade was opened is 8.
- The average number of months that all of a applicant's credit accounts have been open is 66.
- The number of satisfactory trades is 6.
- The number of trades 60+ ever to derogatory public record is 2.
- The number of trades 90+ ever to derogatory public record is 0.
- The percentage of trades that have never been delinquent is 75%.
- The number of months since the most recent delinquency is 45.
- The worst delinquency in the last 12 months is classified as 'unknown delinquency'.
- The worst delinquency ever recorded on this file is classified as '60 days delinquent'.
- The total number of trades (total number of credit accounts) is 8.
- The number of trades opened in the last 12 months is 2.
- The percentage of trades that are installment loans is 75%.
- The number of months since the most recent credit inquiry (excluding duplicates within 7 days) is 0.
- The number of credit inquiries in the last 6 months is 4.
- The number of credit inquiries in the last 6 months (excluding duplicates that are likely due to price comparison shopping within 7 days) is 4.
- The percentage of revolving credit utilization (revolving balance divided by credit limit) is 47%.
- The percentage of installment credit utilization (installment balance divided by original loan amount) is 86%.
- The number of revolving trades with outstanding balance is 1.
- The number of installment trades with outstanding balance is 2.
- The number of bank or national trades with high utilization is 0.
- The percentage of trades that currently carry a balance is 100%.

Here is a example of good credit user profile:
- The external risk estimate is 68, where lower scores indicate higher credit risk.
- The number of months since the oldest trade was opened is 105.
- The number of months since the most recent trade was opened is 2.
- The average number of months that all of a applicant's credit accounts have been open is 62.
- The number of satisfactory trades is 34.
- The number of trades 60+ ever to derogatory public record is 0.
- The number of trades 90+ ever to derogatory public record is 0.
- The percentage of trades that have never been delinquent is 91%.
- The number of months since the most recent delinquency is 17.
- The worst delinquency in the last 12 months is classified as 'unknown delinquency'.
- The worst delinquency ever recorded on this file is classified as '30 days delinquent'.
- The total number of trades (total number of credit accounts) is 57.
- The number of trades opened in the last 12 months is 2.
- The percentage of trades that are installment loans is 37%.
- The number of months since the most recent credit inquiry (excluding duplicates within 7 days) is 14.
- The number of credit inquiries in the last 6 months is 0.
- The number of credit inquiries in the last 6 months (excluding duplicates that are likely due to price comparison shopping within 7 days) is 0.
- The percentage of revolving credit utilization (revolving balance divided by credit limit) is 51%.
- The percentage of installment credit utilization (installment balance divided by original loan amount) is 65%.
- The number of revolving trades with outstanding balance is 5.
- The number of installment trades with outstanding balance is 4.
- The number of bank or national trades with high utilization is 2.
- The percentage of trades that currently carry a balance is 56%.
"""

# https://www.sciencedirect.com/science/article/pii/S2666827024000100
@dataclass
class InformedGPTConfig(BaseConfig):
    use_examples: bool = field(
        default=False,
        metadata={"help": "Whether to include examples in the prompt."},
    )
    examples: str = field(
        default=examples,
        metadata={"help": "The examples to include in the prompt if use_examples is True."},
    )
