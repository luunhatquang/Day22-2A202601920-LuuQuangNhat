# Data Card

- **Dataset name**: Machine Learning & Preference Alignment Sample Dataset (`data/sample_preferences.jsonl`)
- **Source**: Curated ML concept question-answer preference pairs for alignment training and evaluation.
- **License/permission**: MIT / Educational Research Use.
- **Schema**:
  - `prompt` (string, required): Question or instruction in machine learning domains.
  - `chosen` (string, required): High-quality, accurate, and contextually grounded response.
  - `rejected` (string, required): Inaccurate, misleading, or low-quality alternative response.
  - `metadata` (dict, optional): Contains `domain` (e.g., "education") and `rubric` (e.g., "accuracy").
- **Labeling rubric**: Pairwise preference where chosen represents technically factual, comprehensive explanations and rejected represents common misconceptions or flawed definitions.
- **Known biases**: Focused on English machine learning and deep learning conceptual QA.
- **Safety/PII checks**: RegEx-based automated PII detection for email, phone numbers, API keys, and identifiers.
- **Train/validation/test split method**: Prompt-level deterministic grouping and hashing to ensure zero prompt leakage across training and validation splits.
