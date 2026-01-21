# Welcome to the LEXISIGHT Wiki

**LEXISIGHT** is an AI Lawyer Assistant focused on **Indian Contract Law**. This wiki documents the project's architecture, setup, usage, and development details.

## 📚 Documentation Sections

*   **[Setup & Installation](Setup.md)**: Guide to setting up the development environment and running the project locally or on Google Colab.
*   **[Architecture](Architecture.md)**: High-level overview of the system architecture and data pipeline.
*   **[Backend](Backend.md)**: Details about the Python scripts, data collection, and inference API.
*   **[Frontend](Frontend.md)**: Information about the React application and user interface.
*   **[Dataset](Dataset.md)**: Explanation of the `MANISH.json` dataset and data sources.
*   **[Training](Training.md)**: Insights into the LLaMA model fine-tuning process.
*   **[API Reference](API_Reference.md)**: Documentation for the API endpoints.

## 🚀 Project Overview

LEXISIGHT helps users navigate Indian Contract Law disputes by:
1.  **Asking Clarifying Questions**: Gathering necessary facts about the case.
2.  **Providing Legal Analysis**: Offering relevant sections, procedures, strategic advice, and estimated outcomes.

The system uses a fine-tuned **LLaMA-3.2-1B** model and is accessible via a **React** frontend communicating with a **FastAPI** backend.

## 👥 Contributors

*   **Manish Bej** (Author)
*   **A.P. Prithwijit Polley** (Supervisor)
