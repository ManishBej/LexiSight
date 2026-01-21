# Dataset Documentation

The core of LEXISIGHT's legal knowledge comes from the **MANISH.json** dataset. This dataset is curated from Indian contract law judgments and is used to fine-tune the LLaMA model.

## 📄 Dataset Structure

`MANISH.json` is a JSON file containing a list of case objects. Each object represents a single legal case and includes the following fields:

| Field | Description |
| :--- | :--- |
| `case_id` | Unique identifier for the case. |
| `summary` | A concise summary of the case facts. |
| `questions` | A list of clarifying questions that a lawyer might ask to understand the situation better. |
| `relevant_sections` | The specific sections of the Indian Contract Act (or other relevant laws) that apply. |
| `suggested_actions` | Recommended legal procedures or steps the user should take. |
| `strategic_advice` | Advice on strategy, negotiation, or litigation tactics. |
| `estimated_outcome` | A prediction of the likely outcome based on precedents. |

## 🛠️ Creation Process

1.  **Scraping**: The `data collection using selenium.py` script downloads PDF judgments from the Indian Kanoon website.
2.  **Processing**: The `DATASET MAKER.py` script parses these PDFs. It likely uses text extraction (e.g., PyMuPDF) and heuristics or a base LLM to extract the structured fields mentioned above.
3.  **Validation**: The data is checked for consistency before being saved to `MANISH.json`.

## 📊 Statistics

*   **Domain**: Indian Contract Law.
*   **Size**: Approximately 4000+ entries (based on project report notes).
*   **Format**: JSON.
