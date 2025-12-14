
# Uninstall existing torch/torchvision first to avoid conflicts
!pip uninstall -y torch torchvision torchaudio

# Instal[l PyTorch, torchvision, and torchaudio compatible with Colab's CUDA version
!pip install torch torchvision torchaudio

# Install other dependencies AFTER torch/torchvision
!pip install --upgrade transformers datasets sentence-transformers spacy PyMuPDF nltk accelerate bitsandbytes fastapi uvicorn pyngrok
!python -m spacy download en_core_web_sm

# ===============================
# 1. Environment Setup & Mount Google Drive
# ===============================
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import nltk
import torch
import pickle
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from nltk.tokenize import word_tokenize

nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ===============================
# 2. Hugging Face Authentication & Model Setup
# ===============================
# Replace with your actual Hugging Face access token
HF_TOKEN = "PASTE_YOUR_HF_TOKEN_HERE"
os.environ["HF_ACCESS_TOKEN"] = HF_TOKEN

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
# Ensure pad token is set (using EOS token)
tokenizer.pad_token = tokenizer.eos_token
# Load model architecture
model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN)
model.to(device)

# ===============================
# 3. Load Fine-Tuned Model State from .pkl File
# ===============================
pkl_path = "/content/drive/My Drive/llama_finetuned_state_dict.pkl"  # Adjust path as necessary
with open(pkl_path, "rb") as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()  # Set model to evaluation mode
print("✅ Fine-tuned model state loaded from .pkl file.")

import json
import re
from typing import List, Dict, Any

# ===============================
# 4. Load and Process Legal Cases Dataset
# ===============================
def load_legal_cases_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the legal cases dataset from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} legal cases from dataset.")
        return data
    except Exception as e:
        print(f"⚠️ Error loading dataset: {e}")
        # Return empty list as fallback
        return []

# ===============================
# 5. Case Retrieval System
# ===============================
def retrieve_relevant_cases(query: str, case_summary: str, dataset: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the most relevant cases from the dataset based on query and summary."""
    if not dataset:
        print("⚠️ Dataset is empty, using default case summary.")
        return []

    # Simple keyword matching for relevance (a more sophisticated approach would use embeddings)
    keywords = set(re.findall(r'\b\w+\b', query.lower() + " " + case_summary.lower()))

    # Score cases based on keyword matches
    scored_cases = []
    for case in dataset:
        if not case or "summary" not in case:
            continue

        case_text = case.get("summary", "").lower()
        score = sum(1 for keyword in keywords if keyword in case_text)

        # Bonus points for cases with relevant sections and questions
        if "relevant_sections" in case and case["relevant_sections"]:
            score += 5
        if "questions" in case and case["questions"]:
            score += 3

        scored_cases.append((score, case))

    # Sort by score and return top_k cases
    scored_cases.sort(reverse=True, key=lambda x: x[0])
    return [case for _, case in scored_cases[:top_k]]

# Function to extract key insights from relevant cases
def extract_case_insights(relevant_cases: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Extract structured insights from the relevant cases."""
    insights = {
        "relevant_sections": set(),
        "strategic_advice": [],
        "suggested_actions": []
    }

    for case in relevant_cases:
        if "relevant_sections" in case and case["relevant_sections"]:
            for section in case["relevant_sections"]:
                insights["relevant_sections"].add(section)

        if "strategic_advice" in case and case["strategic_advice"]:
            insights["strategic_advice"].extend(case["strategic_advice"][:2])  # Limit to top 2

        if "suggested_actions" in case and case["suggested_actions"]:
            insights["suggested_actions"].extend(case["suggested_actions"][:2])  # Limit to top 2

    # Convert set to list for relevant_sections
    insights["relevant_sections"] = list(insights["relevant_sections"])

    # Deduplicate and limit lists
    for key in ["strategic_advice", "suggested_actions"]:
        insights[key] = list(set(insights[key]))[:4]  # Keep only top 4 unique items

    return insights

# ===============================
# 6. Inference Function (updated)
# ===============================
def generate_text(prompt: str, max_length: int = 128000) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128000,
            num_beams=8,
            repetition_penalty=3.0,
            no_repeat_ngram_size=4,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===============================
# 7. Build Prompt Functions with Dataset Integration
# ===============================
import re
def build_prompt_part1(query: str, case_summary: str, relevant_cases: List[Dict[str, Any]]) -> str:
    # Extract questions from relevant cases to guide the model
    example_questions = []
    for case in relevant_cases:
        if "questions" in case and case["questions"]:
            example_questions.extend(case["questions"][:2])  # Take up to 2 questions per case

    # Format example questions for prompt
    example_questions_text = ""
    if example_questions:
        example_questions_text = "Here are some examples of relevant questions from similar cases:\n"
        example_questions_text += "\n".join(example_questions[:4])  # Limit to 4 examples
        example_questions_text += "\n\n"

    prompt = (
        "You are a legal assistant specializing in contract law. Your task is to generate 4 unique clarifying questions "
        "about the user's contract breach situation.\n\n"
        f"{example_questions_text}"
        "IMPORTANT INSTRUCTIONS:\n"
        "1. Each question must focus on gathering facts about the specific contract breach described\n"
        "2. Focus on delivery timelines, documentation, evidence of losses, and prior communication\n"
        "3. Number your questions 1-4\n"
        "4. Each question must end with a question mark\n"
        "5. Do not repeat or overlap in the questions\n"
        "6. Keep each question concise and direct\n\n"
        "User Query:\n" + query + "\n\n"
        "Case Summary:\n" + case_summary + "\n\n"
        "Your 4 clarifying questions:"
    )
    return prompt

def build_prompt_part2(query: str, case_summary: str, clarifying_answers: str, relevant_cases: List[Dict[str, Any]]) -> str:
    # Extract insights from relevant cases
    insights = extract_case_insights(relevant_cases)

    # Format relevant sections
    sections_text = ""
    if insights["relevant_sections"]:
        sections_text = "Potentially relevant legal sections:\n"
        sections_text += "\n".join([f"- {section}" for section in insights["relevant_sections"]])
        sections_text += "\n\n"

    # Format strategic advice
    advice_text = ""
    if insights["strategic_advice"]:
        advice_text = "Strategic considerations from similar cases:\n"
        advice_text += "\n".join([f"- {advice}" for advice in insights["strategic_advice"]])
        advice_text += "\n\n"

    prompt = (
        "You are a legal assistant specializing in contract law. Based on the information below, provide a detailed legal analysis "
        "for a case involving a breach of contract due to late delivery.\n\n"
        f"{sections_text}"
        f"{advice_text}"
        "FORMAT YOUR RESPONSE WITH EXACTLY THESE FOUR SECTIONS AND HEADINGS:\n\n"
        "a) Relevant Legal Sections and Articles: [Identify specific laws and precedents applicable to this case]\n\n"
        "b) Suggested Legal Procedures: [List specific actions the client should take]\n\n"
        "c) Strategic Advice: [Provide practical recommendations]\n\n"
        "d) Estimated Outcome: [Analyze the likely result based on similar cases]\n\n"
        "Case Information:\n"
        "User Query: " + query + "\n\n"
        "Case Summary: " + case_summary + "\n\n"
        "Answers to Clarifying Questions:\n" + clarifying_answers + "\n\n"
        "BEGIN YOUR ANALYSIS:"
    )
    return prompt

# Function to clean model output and extract just the generated content
def clean_model_output(output, is_part1=False):
    if is_part1:
        # First, remove any instances of the original query that might be in the output
        lines = output.split('\n')
        filtered_lines = [l for l in lines if not l.startswith("I signed a contract") and "We agreed on a delivery date" not in l]
        output = '\n'.join(filtered_lines)

        # Extract questions that end with question marks
        questions = []
        for line in output.split('\n'):
            line = line.strip()
            # Extract actual questions ending with question marks
            if '?' in line:
                # Split by number prefix if it exists
                parts = re.split(r'^\d+\.?\s*', line)
                # Take the part after the number or the whole line
                question = parts[-1].strip()
                if question.endswith('?'):
                    questions.append(question)

        # Take exactly 4 questions and format them properly
        formatted_questions = []
        for i, q in enumerate(questions[:4]):
            formatted_questions.append(f"{i+1}. {q}")

        return '\n'.join(formatted_questions)
    else:  # Part 2 processing
        # Define the expected sections in the correct order
        sections = [
            "a) Relevant Legal Sections and Articles",
            "b) Suggested Legal Procedures",
            "c) Strategic Advice",
            "d) Estimated Outcome"
        ]

        section_content = {}

        # First, try to find each section with exact heading match
        current_section = None
        current_content = []

        for line in output.split('\n'):
            # Check if line starts a new section
            found_section = False
            for section in sections:
                if section.lower() in line.lower() or section.lower().split(':')[0] in line.lower():
                    if current_section:
                        section_content[current_section] = '\n'.join(current_content).strip()
                    current_section = section
                    current_content = []
                    found_section = True
                    break

            if not found_section and current_section:
                current_content.append(line)

        # Add last section
        if current_section and current_content:
            section_content[current_section] = '\n'.join(current_content).strip()

        # If sections are missing, try identifying by keywords
        keywords = {
            "a) Relevant Legal Sections and Articles": ["legal section", "relevant law", "applicable law", "statute"],
            "b) Suggested Legal Procedures": ["procedure", "action", "steps", "process", "suggested action"],
            "c) Strategic Advice": ["strategic", "advice", "recommendation", "consider"],
            "d) Estimated Outcome": ["outcome", "result", "likely", "probability", "chance"]
        }

        for section in sections:
            if section not in section_content:
                for line in output.split('\n'):
                    for keyword in keywords[section]:
                        if keyword.lower() in line.lower() and ":" in line:
                            potential_section = line.split(":")[0].strip()
                            rest_of_text = ':'.join(line.split(':')[1:]).strip()

                            section_text = ""
                            capturing = False
                            for inner_line in output.split('\n'):
                                if potential_section in inner_line:
                                    capturing = True
                                    # Skip the header line
                                    continue
                                elif capturing and any(s.lower() in inner_line.lower() for s in sections if s != section):
                                    break
                                elif capturing:
                                    section_text += inner_line + '\n'

                            if not section_text.strip() and rest_of_text:
                                section_text = rest_of_text

                            if section_text.strip():
                                section_content[section] = section_text.strip()
                                break

        # Build final structured output with proper line breaks
        result = []
        for section in sections:
            content = section_content.get(section, "No information provided.")
            result.append(f"{section}:")
            result.append(content)
            result.append("\n")  # Add an extra empty line between sections

        return "\n".join(result)

# Install the necessary packages if not already installed
!pip install fastapi uvicorn pyngrok

# ===============================
# 8. FastAPI and Pydantic Models for API
# ===============================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok



# Define data models for API requests and responses
class QueryRequest(BaseModel):
    query: str

class AnswersRequest(BaseModel):
    query: str
    case_summary: str
    answers: List[str]

class Part1Response(BaseModel):
    questions: List[str]
    case_summary: str

class Part2Response(BaseModel):
    relevant_legal_sections: str
    suggested_legal_procedures: str
    strategic_advice: str
    estimated_outcome: str

    class Config:
        # Ensure JSON schema properly represents multi-line strings
        schema_extra = {
            "example": {
                "relevant_legal_sections": "Section 1\n\nSection 2",
                "suggested_legal_procedures": "Step 1\n\nStep 2",
                "strategic_advice": "Advice 1\n\nAdvice 2",
                "estimated_outcome": "Outcome details\n\nMore details"
            }
        }

# Create FastAPI app
app = FastAPI(title="Legal Assistant API")

# Add CORS middleware to allow cross-origin requests (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for dataset
global_legal_cases = []

@app.on_event("startup")
async def startup_event():
    global global_legal_cases
    # Load legal cases dataset
    dataset_path = "/content/drive/My Drive/MANISH.json"
    global_legal_cases = load_legal_cases_dataset(dataset_path)
    print(f"✅ API loaded {len(global_legal_cases)} legal cases from dataset.")

# API endpoints
@app.post("/generate_part1", response_model=Part1Response)
async def generate_part1(request: QueryRequest):
    try:
        # Retrieve relevant cases using existing keyword matching logic
        keywords = set(re.findall(r'\b\w+\b', request.query.lower()))
        scored_cases = []
        for case in global_legal_cases:
            if not case or "summary" not in case:
                continue

            case_text = case.get("summary", "").lower()
            score = sum(1 for keyword in keywords if keyword in case_text)

            # Bonus points for cases with relevant sections and questions
            if "relevant_sections" in case and case["relevant_sections"]:
                score += 5
            if "questions" in case and case["questions"]:
                score += 3

            scored_cases.append((score, case))

        # Sort by score and get top cases
        scored_cases.sort(reverse=True, key=lambda x: x[0])
        relevant_cases = [case for _, case in scored_cases[:3]]  # Get top 3 cases

        # Generate case summary using the same logic from the pipeline
        if relevant_cases:
            case_summary = "This case involves a breach of contract where a vendor delivered materials late. "
            if "summary" in relevant_cases[0]:
                top_case = relevant_cases[0]["summary"]
                excerpt = top_case[:15000] + "..." if len(top_case) > 15000 else top_case
                case_summary += f"Similar precedent indicates: {excerpt}"
        else:
            # Fallback summary
            case_summary = (
                "This case involves a breach of contract where a vendor delivered materials late. "
                "The contract did not specify penalties for delays, raising questions about damages."
            )

        # Generate clarifying questions using existing functions
        prompt_part1 = build_prompt_part1(request.query, case_summary, relevant_cases)
        part1_response = generate_text(prompt_part1)
        cleaned_output = clean_model_output(part1_response, is_part1=True)

        # Extract questions as a list
        questions = [q.strip() for q in cleaned_output.split('\n') if q.strip()]
        # Ensure we have exactly 4 questions and remove numbering
        formatted_questions = []
        for i, q in enumerate(questions[:4]):
            # Remove numbering if present
            question = re.sub(r'^\d+\.\s*', '', q)
            formatted_questions.append(question)

        # Fill with default questions if needed
        while len(formatted_questions) < 4:
            formatted_questions.append(f"Could you provide more details about the contract situation?")

        return Part1Response(questions=formatted_questions[:4], case_summary=case_summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.post("/generate_part2", response_model=Part2Response)
async def generate_part2(request: AnswersRequest):
    try:
        # Retrieve relevant cases again (ensuring consistency)
        keywords = set(re.findall(r'\b\w+\b', request.query.lower()))
        scored_cases = []
        for case in global_legal_cases:
            if not case or "summary" not in case:
                continue
            case_text = case.get("summary", "").lower()
            score = sum(1 for keyword in keywords if keyword in case_text)
            if "relevant_sections" in case and case["relevant_sections"]:
                score += 5
            if "questions" in case and case["questions"]:
                score += 3
            scored_cases.append((score, case))

        scored_cases.sort(reverse=True, key=lambda x: x[0])
        relevant_cases = [case for _, case in scored_cases[:3]]

        # Format the answers for the prompt
        clarifying_answers = "\n".join([f"Answer {i+1}: {a}" for i, a in enumerate(request.answers)])

        # Generate final legal analysis
        prompt_part2 = build_prompt_part2(request.query, request.case_summary, clarifying_answers, relevant_cases)
        part2_response = generate_text(prompt_part2)

        # Clean and extract the structured response
        structured_output = clean_model_output(part2_response)

        # Extract the sections
        sections = {
            "a) Relevant Legal Sections and Articles": "",
            "b) Suggested Legal Procedures": "",
            "c) Strategic Advice": "",
            "d) Estimated Outcome": ""
        }

        # Parse the response to extract each section
        for section in sections:
            pattern = re.escape(section) + r":(.*?)(?=" + "|".join([re.escape(s) + ":" for s in sections if s != section]) + "|$)"
            match = re.search(pattern, structured_output, re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()

        # Process the sections with explicit paragraph separations
        processed_sections = {}
        for key, content in sections.items():
            # Split content by sentences or line breaks
            paragraphs = re.split(r'(?<=[.!?])\s+|\n+', content)
            # Filter out empty paragraphs
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            # Join with explicit double line breaks
            processed_sections[key] = "\n\n".join(paragraphs)

            # Special handling for numbered lists to ensure they're properly formatted
            if re.search(r'^\d+\.', processed_sections[key], re.MULTILINE):
                # Find all numbered items and ensure they have double line breaks before them
                processed_sections[key] = re.sub(r'(?<!\n\n)(\d+\.)', r'\n\n\1', processed_sections[key])
                processed_sections[key] = re.sub(r'\n{3,}', '\n\n', processed_sections[key])

        return Part2Response(
            relevant_legal_sections=processed_sections["a) Relevant Legal Sections and Articles"],
            suggested_legal_procedures=processed_sections["b) Suggested Legal Procedures"],
            strategic_advice=processed_sections["c) Strategic Advice"],
            estimated_outcome=processed_sections["d) Estimated Outcome"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")

@app.post("/full_analysis")
async def full_analysis(request: QueryRequest):
    """Combined endpoint for a complete analysis simulation"""
    # First generate questions (Part 1)
    part1 = await generate_part1(request)

    # Simulate user answers (for testing)
    mock_answers = ["The contract had a delivery date specified but no penalties.",
                   "I documented all communications regarding the delay.",
                   "The delay cost me approximately $5,000 in project overruns.",
                   "The vendor acknowledged the delay but claimed it was due to supply issues."]

    # Generate final analysis (Part 2)
    answers_request = AnswersRequest(
        query=request.query,
        case_summary=part1.case_summary,
        answers=mock_answers
    )
    part2 = await generate_part2(answers_request)

    # Return combined results
    return {
        "part1": part1,
        "part2": part2
    }

@app.get("/")
async def root():
    return {"message": "Legal Assistant API is running"}

# ===============================
# 11. Setup ngrok and run the API server
# ===============================
import nest_asyncio
nest_asyncio.apply() # Apply nest_asyncio patch

def start_api_server():
    # Set up ngrok tunnel
    ngrok.set_auth_token("2vj1nA1eHTDyPNUzuvil8uldP5O_2McvnkbRQ4bpjCEH85X9M") # Add this line with your authtoken
    ngrok_tunnel = ngrok.connect(8000)
    print(f"✅ Public URL: {ngrok_tunnel.public_url}")
    print(f"⚠️ Note: This URL will expire when this Colab session ends")
    print(f"🔗 Use this URL in your React frontend's API calls")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ===============================
# 12. Main execution logic - Replaced with API server
# ===============================
if __name__ == "__main__":
    # Comment out the original pipeline run
    # run_two_phase_pipeline()

    # Start the API server instead
    start_api_server()