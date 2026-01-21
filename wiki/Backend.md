# Backend Documentation

The backend is built with Python and FastAPI. It handles data processing, model training, and serving the AI model.

## 📂 File Structure

*   `backend/`
    *   `data collection using selenium.py`: Web scraper for court judgments.
    *   `DATASET MAKER.py`: Processes raw text into the `MANISH.json` dataset.
    *   `llama training & finetuning program.py`: Script to fine-tune the LLaMA model.
    *   `FInal AI using trained & finetuned llama.py`: The main inference API server.
    *   `Evaluate & test accuracy of AI output.py`: Evaluation script.
    *   `post process.py`: Utilities for post-processing model outputs.
    *   `Prompt & mislenious.txt`: Miscellaneous notes and prompts.

## 🐍 Key Scripts

### Inference API (`FInal AI using trained & finetuned llama.py`)

This script launches a FastAPI server. It performs the following:
1.  Loads the fine-tuned LLaMA model from `llama_finetuned_state_dict.pkl`.
2.  Initializes the tokenizer and model configuration.
3.  Sets up API endpoints (`/generate_part1`, `/generate_part2`).
4.  Starts an ngrok tunnel to expose the local server.

**Key Libraries**: `fastapi`, `uvicorn`, `transformers`, `torch`, `pyngrok`.

### Data Collection (`data collection using selenium.py`)

This script uses Selenium to automate the downloading of legal case PDFs from Indian Kanoon.
*   **Target**: Contract law judgments.
*   **Output**: PDF files stored locally.

### Dataset Maker (`DATASET MAKER.py`)

Converts raw PDF content into a structured JSON dataset.
*   **Input**: PDF files.
*   **Processing**: Text extraction, cleaning, and segmentation.
*   **Output**: `MANISH.json` containing fields like `summary`, `questions`, `relevant_sections`.

## 🛠️ Dependencies

Ensure the following packages are installed:
```bash
transformers
torch
sentence-transformers
fastapi
uvicorn
pyngrok
flask
accelerate
bitsandbytes
```
Also requires `spacy` model `en_core_web_sm`:
```bash
python -m spacy download en_core_web_sm
```
