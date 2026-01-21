# Setup and Installation

This guide covers how to set up and run LEXISIGHT. You can run the backend in Google Colab (recommended for development and demos) or locally.

## ☁️ Running in Google Colab (Recommended)

This method is easiest as it leverages free GPU resources for the LLaMA model.

### Prerequisites
*   Google Account
*   Hugging Face Token
*   ngrok Authtoken

### Steps
1.  **Open the Notebook**: Open `FInal AI using trained & finetuned llama.py` (or the provided Colab notebook).
2.  **Mount Google Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  **Set Environment Variables**:
    Set your tokens in the notebook (do not commit these to version control).
    ```python
    %env HF_TOKEN=hf_...
    %env NGROK_AUTHTOKEN=...
    ```
4.  **Install Dependencies**:
    ```bash
    !pip install -q transformers torch sentence-transformers fastapi uvicorn pyngrok flask accelerate bitsandbytes
    !python -m spacy download en_core_web_sm
    ```
5.  **Place Required Files**:
    Upload `MANISH.json` and `llama_finetuned_state_dict.pkl` to your Google Drive path referenced by the notebook.
6.  **Start the Server**:
    Run the cell that starts the FastAPI server. It will output a public ngrok URL.
7.  **Connect Frontend**:
    Update the API base URL in the frontend code with the ngrok URL.

## 💻 Running Locally

### Prerequisites
*   Python 3.9+
*   Node.js and npm
*   Git

### Backend Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/<your-org>/lexisight.git
    cd lexisight
    ```
2.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `requirements.txt` is missing, install the following packages manually:*
    ```bash
    pip install transformers torch sentence-transformers fastapi uvicorn pyngrok flask accelerate bitsandbytes
    ```
4.  **Place Model Files**:
    Place `llama_finetuned_state_dict.pkl` and `MANISH.json` in the `backend/DATASET` or appropriate artifact directory.
5.  **Set Environment Variables**:
    ```bash
    export HF_TOKEN="hf_..."
    export NGROK_AUTHTOKEN="..."
    ```
6.  **Start Server**:
    ```bash
    python backend/"FInal AI using trained & finetuned llama.py"
    # or
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ```

### Frontend Setup

1.  **Navigate to Frontend Directory**:
    ```bash
    cd frontend
    ```
2.  **Install Dependencies**:
    ```bash
    npm install
    ```
3.  **Start React App**:
    ```bash
    npm start
    ```
    The app will open at `http://localhost:3000`.

## ⚠️ Important Notes

*   **GPU Requirement**: Running LLaMA locally requires significant GPU memory. If you don't have a GPU, use Google Colab.
*   **API URL**: Every time you restart the Colab/ngrok session, the API URL changes. You must update the frontend to point to the new URL.
