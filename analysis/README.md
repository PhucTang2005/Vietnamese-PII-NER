# Error Analysis

This folder contains summarized error analysis artifacts for the Vietnamese PII NER project.

The goal is to keep the public GitHub repository clean while still showing that the models were analyzed beyond aggregate F1 scores. The full internal analysis scripts and very large token-level outputs are intentionally not included.

## Included Files

| File | Purpose |
|---|---|
| `PII_NER_Error_Analysis_Report_Clean.pdf` | Concise human-readable report for README, CV review, and presentation preparation |
| `model_error_summary.csv` | Sample-level mean F1 and exact-match summary for each model |
| `error_type_breakdown.csv` | Token-level error distribution by model |
| `entity_type_report.csv` | Per-entity precision, recall, F1, and support |
| `difficult_entity_types.csv` | Entity types with the lowest average F1 across models |
| `case_studies_sample.csv` | Small set of representative examples for qualitative explanation |

## Main Findings

- Overall ranking: **PhoBERT ≈ XLM-RoBERTa > BiLSTM-CRF > BiLSTM**.
- BiLSTM-CRF improves over BiLSTM because CRF encourages more consistent BIO label transitions.
- The largest remaining error group is **wrong entity type**, especially between PII types with similar surface patterns.
- Difficult entity groups include `DOB`, `DATE`, `JOBTYPE`, `MASKEDNUMBER`, and `CREDITCARDNUMBER`.
- PhoBERT and XLM-RoBERTa are very close, suggesting that Vietnamese PII NER depends on both Vietnamese context and universal PII patterns.

## Not Included

The following files are kept internal to avoid making the repository unnecessarily large or difficult to navigate:

- `analysis_utils.py`
- `error_analysis.py`
- full `token_level_analysis.csv`
- full `error_summary.csv`
- full `hard_entity_samples_detailed.csv`
