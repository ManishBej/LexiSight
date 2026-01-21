# Model Training

LEXISIGHT uses a fine-tuned version of **LLaMA-3.2-1B** to generate legal analysis.

## 🏋️ Training Process

The training is handled by the script `llama training & finetuning program.py`.

### 1. Data Preparation
The `MANISH.json` dataset is loaded and converted into prompt-completion pairs.
*   **Prompt**: Context + User Query
*   **Completion**: The structured analysis (sections, advice, outcome).

### 2. Tokenization
The text data is tokenized using the LLaMA tokenizer. Padding and truncation are applied to fit the model's context window.

### 3. Fine-Tuning
*   **Base Model**: `meta-llama/Llama-3.2-1B`
*   **Framework**: Hugging Face `transformers` (Trainer API).
*   **Environment**: Google Colab (GPU is essential).
*   **Method**: Full fine-tuning or PEFT/LoRA (depending on configuration).

### 4. Output
The trained model weights are saved as a dictionary: `llama_finetuned_state_dict.pkl`.

## ⚙️ Hyperparameters

(Typical values, check script for exact settings)
*   **Batch Size**: 2 (per device)
*   **Epochs**: 3
*   **Learning Rate**: 2e-5
*   **Optimizer**: AdamW

## 📝 Reproduction

To retrain the model:
1.  Ensure `MANISH.json` is available.
2.  Open `llama training & finetuning program.py` in Colab.
3.  Set your Hugging Face token.
4.  Run the training cells.
5.  Download the resulting `.pkl` file.
