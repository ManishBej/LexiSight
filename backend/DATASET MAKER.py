# Install required system dependencies
!apt-get update
!apt-get install -y poppler-utils tesseract-ocr

# Install Python packages
!pip install pytesseract
!pip install pymupdf
!pip install pdf2image
!pip install transformers
!pip install sentence-transformers
!pip install tqdm
!pip install dataclasses

# These are typically pre-installed in Colab, but included for completeness
!pip install torch
!pip install numpy
!pip install pandas

!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


import os
import json
import torch
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import re
import gc
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# PDF processing
import fitz  # PyMuPDF
from pdf2image import convert_from_path

# Machine learning
from transformers import (
    LayoutLMv2Processor, 
    LayoutLMv2ForSequenceClassification,
    LayoutLMv2ForTokenClassification,
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModel
)
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CaseEntry:
    """Data structure for a single contract case entry."""
    case_id: str
    scenario: str
    questions: List[str]
    relevant_sections: List[str]
    suggested_actions: List[str]
    strategic_advice: List[str]
    estimated_outcome: str
    scenario_embedding: List[float]

class HardwareManager:
    """Utility class to detect and manage hardware resources."""
    
    @staticmethod
    def detect_hardware():
        """Detect available hardware resources."""
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
        gpu_memory = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count)] if gpu_available else []
        cpu_count = os.cpu_count()
        
        logger.info(f"GPU Available: {gpu_available}")
        logger.info(f"GPU Count: {gpu_count}")
        logger.info(f"GPU Names: {gpu_names}")
        logger.info(f"GPU Memory (GB): {gpu_memory}")
        logger.info(f"CPU Count: {cpu_count}")
        
        return {
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "gpu_memory": gpu_memory,
            "cpu_count": cpu_count
        }
    
    @staticmethod
    def calculate_optimal_workers(hardware_info, max_resource_usage=0.95):
        """Calculate optimal number of workers based on available hardware."""
        if hardware_info["gpu_available"]:
            # For GPU: Typically 1-2 workers per GPU depending on memory
            # Estimating memory needs: LayoutLMv2-large ~1.2GB, T5-large ~2.7GB
            # Plus memory for PDF processing, roughly 5GB per worker to be safe
            total_memory = sum(hardware_info["gpu_memory"])
            memory_per_worker = 5.0  # GB
            gpu_workers = max(1, int(total_memory * max_resource_usage / memory_per_worker))
            
            # Use fewer CPU workers when GPU is available to prevent CPU bottlenecks
            cpu_workers = min(hardware_info["cpu_count"] - 1, gpu_workers * 2)
            return min(gpu_workers, cpu_workers)
        else:
            # For CPU-only: Use most cores but leave some for system
            return max(1, int(hardware_info["cpu_count"] * max_resource_usage))

from google.colab import auth
from oauth2client.client import GoogleCredentials
import googleapiclient.discovery
import tempfile
import shutil
import io
from googleapiclient.http import MediaIoBaseDownload

class GoogleDriveHandler:
    """Handler for interacting with Google Drive using the API."""
    
    @staticmethod
    def mount_drive():
        """Mount Google Drive."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully.")
            return True
        except ImportError:
            logger.warning("Not running in Google Colab. Skipping drive mounting.")
            return False

    @staticmethod
    def get_drive_service():
        """Get authenticated Google Drive service."""
        try:
            auth.authenticate_user()
            creds = GoogleCredentials.get_application_default()
            drive_service = googleapiclient.discovery.build('drive', 'v3', credentials=creds)
            return drive_service
        except Exception as e:
            logger.error(f"Failed to initialize Drive service: {e}")
            return None
    
    @staticmethod
    def find_folder_id(folder_name, drive_service=None):
        """Find Google Drive folder ID by name."""
        if drive_service is None:
            drive_service = GoogleDriveHandler.get_drive_service()
            if drive_service is None:
                return None
        
        results = drive_service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        items = results.get('files', [])
        if items:
            logger.info(f"Found folder '{folder_name}' with ID: {items[0]['id']}")
            return items[0]['id']
        else:
            logger.warning(f"Folder '{folder_name}' not found in Google Drive")
            return None
    
    @staticmethod
    def scan_for_pdfs(root_folder: str) -> List[str]:
        """
        Scan a Google Drive folder for PDF files using the Drive API.
        
        Args:
            root_folder: Name or path of the folder to scan
            
        Returns:
            List of file IDs and names of PDF files
        """
        # For Drive API, we need the folder ID, not the path
        folder_name = os.path.basename(root_folder)
        drive_service = GoogleDriveHandler.get_drive_service()
        if drive_service is None:
            logger.error("Failed to initialize Drive service")
            return []
        
        # Find the folder ID
        folder_id = GoogleDriveHandler.find_folder_id(folder_name, drive_service)
        if not folder_id:
            logger.error(f"Folder not found: {folder_name}")
            return []
        
        # List files in the folder
        pdf_files = []
        logger.info(f"Scanning for PDF files in folder '{folder_name}'")
        
        try:
            results = drive_service.files().list(
                q=f"'{folder_id}' in parents and mimeType contains 'application/pdf'",
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                for item in tqdm(items, desc="Listing PDFs"):
                    # Store file ID and name in format useful for later
                    pdf_files.append({
                        'id': item['id'],
                        'name': item['name'],
                        'path': f"gdrive://{item['id']}/{item['name']}"  # Custom format to indicate Drive file
                    })
                
                logger.info(f"Found {len(pdf_files)} PDF files")
            else:
                logger.warning(f"No PDF files found in folder '{folder_name}'")
            
            return pdf_files
        except Exception as e:
            logger.error(f"Error scanning for PDFs in Google Drive: {str(e)}")
            return []
    
    @staticmethod
    def download_pdf(file_info, temp_dir=None):
        """
        Download a PDF file from Google Drive.
        
        Args:
            file_info: Dictionary with file 'id' and 'name'
            temp_dir: Directory to save the file (defaults to a temporary directory)
            
        Returns:
            Path to the downloaded file
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        drive_service = GoogleDriveHandler.get_drive_service()
        if drive_service is None:
            logger.error("Failed to initialize Drive service")
            return None
        
        file_id = file_info['id']
        file_name = file_info['name']
        local_path = os.path.join(temp_dir, file_name)
        
        try:
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            logger.info(f"Downloaded {file_name}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading file {file_name}: {str(e)}")
            return None

class ModelManager:
    """Manager for the document understanding and summarization models."""
    
    def __init__(self, use_gpu: bool):
        """
        Initialize the models.
        
        Args:
            use_gpu: Whether to use GPU for inference
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        logger.info("Loading LayoutLMv2 model...")
        self.layout_processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-large-uncased")
        self.layout_model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-large-uncased")
        self.layout_model.to(self.device)
        
        logger.info("Loading T5 model...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        self.t5_model.to(self.device)
        
        logger.info("Loading sentence transformer for embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model.to(self.device)
    
    def extract_layout_info(self, image: Image.Image, text: str) -> Dict[str, Any]:
        """
        Extract layout information using LayoutLMv2.
        
        Args:
            image: PIL Image of a PDF page
            text: Text extracted from the PDF page
            
        Returns:
            Dictionary containing extracted layout information
        """
        try:
            # Limit text length to prevent token overflow
            # A rough estimate is ~4 chars per token on average
            max_chars = 1600  # ~400 tokens to leave room for other inputs
            if len(text) > max_chars:
                logger.info(f"Truncating text from {len(text)} to {max_chars} characters")
                text = text[:max_chars]
            
            # Prepare input for LayoutLMv2
            encoding = self.layout_processor(
                image,
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
            # Move input to device
            for key in encoding.keys():
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].to(self.device)
            
            # Get model output
            with torch.no_grad():
                outputs = self.layout_model(**encoding)
            
            # Process outputs to extract layout information
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            
            # Extract sections, headings, etc. based on predictions
            return {
                "sections": self._extract_sections(text, predictions[0].cpu().numpy(), encoding),
            }
        
        except Exception as e:
            logger.error(f"Error in extract_layout_info: {str(e)}")
            # Return empty sections as fallback
            return {"sections": self._extract_sections(text, None, None)}
    
    def _extract_sections(self, text: str, predictions: np.ndarray, encoding) -> List[str]:
        """Extract document sections based on layout predictions."""
        # This is a placeholder implementation
        # In practice, you would use the token classification results to identify section boundaries
        
        # Simple heuristic: split by common section headers in legal documents
        section_markers = [
            "SECTION", "Article", "CLAUSE", "AGREEMENT", "TERMS", "CONDITIONS",
            "WHEREAS", "NOW, THEREFORE", "IN WITNESS WHEREOF"
        ]
        
        sections = []
        for marker in section_markers:
            pattern = f"({marker}\\s+\\d+|{marker}\\s*:)"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start_idx = match.start()
                section_text = text[start_idx:start_idx + 500]  # Extract ~500 chars after each section marker
                sections.append(section_text.strip())
        
        # Ensure we have at least some sections
        if not sections:
            # Fallback: just extract some chunks from the document
            chunks = [text[i:i+500].strip() for i in range(0, len(text), 500)]
            sections = [chunk for chunk in chunks if len(chunk) > 100][:5]  # Take up to 5 substantial chunks
            
        return sections
    
    def summarize_with_t5(self, text: str, prompt_prefix: str, max_length: int = 150) -> str:
        """
        Summarize text using FLAN-T5.
        
        Args:
            text: Text to summarize
            prompt_prefix: Prefix for the prompt (e.g., "Summarize: ")
            max_length: Maximum length of the generated summary
            
        Returns:
            Summarized text
        """
        # Truncate input if too long
        max_input_length = 1024  # T5 input limit
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        # Prepare input
        input_text = f"{prompt_prefix} {text}"
        input_ids = self.t5_tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        # Generate summary
        with torch.no_grad():
            output = self.t5_model.generate(
                input_ids, 
                max_length=max_length, 
                num_beams=4, 
                early_stopping=True
            )
        
        summary = self.t5_tokenizer.decode(output[0], skip_special_tokens=True)
        return summary
    
    def extract_questions(self, text: str, max_questions: int = 3) -> List[str]:
        """Extract the most important questions from text."""
        prompt = f"Extract the {max_questions} most important legal questions from this case: {text}"
        
        input_ids = self.t5_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            output = self.t5_model.generate(
                input_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
        
        questions_text = self.t5_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parse questions into a list
        questions = []
        for line in questions_text.split("\n"):
            line = line.strip()
            if line and (line.endswith("?") or "?" in line):
                # Add numbering if not present
                if not re.match(r"^\d+\.", line):
                    questions.append(f"{len(questions)+1}. {line}")
                else:
                    questions.append(line)
        
        # Ensure we have the right number of questions
        if len(questions) < max_questions:
            # Generate additional questions if needed
            additional_prompt = f"Generate {max_questions - len(questions)} more critical legal questions about this case: {text}"
            input_ids = self.t5_tokenizer(additional_prompt, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                output = self.t5_model.generate(
                    input_ids,
                    max_length=200,
                    num_beams=4,
                    early_stopping=True
                )
            
            additional_text = self.t5_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Add new questions
            for line in additional_text.split("\n"):
                line = line.strip()
                if line and (line.endswith("?") or "?" in line):
                    if len(questions) < max_questions:
                        if not re.match(r"^\d+\.", line):
                            questions.append(f"{len(questions)+1}. {line}")
                        else:
                            questions.append(line)
        
        # Limit to max_questions
        return questions[:max_questions]
    
    def generate_suggested_actions(self, text: str, num_actions: int = 3) -> List[str]:
        """Generate suggested legal actions based on the case."""
        prompt = f"Generate {num_actions} recommended legal actions for this contract case: {text}"
        
        input_ids = self.t5_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            output = self.t5_model.generate(
                input_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
        
        actions_text = self.t5_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parse actions into list
        actions = []
        for line in actions_text.split("\n"):
            line = line.strip()
            if line:
                # Clean up any numbering or bullet points
                line = re.sub(r"^\d+\.\s*", "", line)
                line = re.sub(r"^-\s*", "", line)
                if line:
                    actions.append(line)
        
        # Ensure we have the right number of actions
        while len(actions) < num_actions:
            actions.append(f"Review all contractual terms and seek legal counsel")
        
        return actions[:num_actions]
    
    def generate_strategic_advice(self, text: str, num_points: int = 2) -> List[str]:
        """Generate strategic legal advice for the case."""
        prompt = f"Provide {num_points} points of concise strategic legal advice for this contract case: {text}"
        
        input_ids = self.t5_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            output = self.t5_model.generate(
                input_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
        
        advice_text = self.t5_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parse advice into list
        advice = []
        for line in advice_text.split("\n"):
            line = line.strip()
            if line:
                # Clean up any numbering or bullet points
                line = re.sub(r"^\d+\.\s*", "", line)
                line = re.sub(r"^-\s*", "", line)
                if line:
                    advice.append(line)
        
        # Ensure we have the right number of advice points
        while len(advice) < num_points:
            advice.append("Consider all legal implications before proceeding")
        
        return advice[:num_points]
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a text."""
        with torch.no_grad():
            embedding = self.embedding_model.encode(text)
        
        return embedding.tolist()

class EntityAnonymizer:
    """Identifies and anonymizes names of parties and entities in legal documents."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.entity_map = {}  # Maps original entities to anonymized versions
        self.entity_types = {
            "plaintiff": [],    # List of entities identified as plaintiffs
            "defendant": [],    # List of entities identified as defendants
            "judge": [],        # List of judges
            "attorney": [],     # List of attorneys
            "witness": [],      # List of witnesses
            "organization": [], # List of organizations
            "other": []         # Other entities
        }
        
        # Common legal terms used to identify entity roles
        self.legal_role_patterns = {
            "plaintiff": [r"plaintiff", r"claimant", r"petitioner", r"complainant", r"appellant"],
            "defendant": [r"defendant", r"respondent", r"accused", r"appellee"],
            "judge": [r"judge", r"justice", r"magistrate", r"honor", r"honourable", r"honorable"],
            "attorney": [r"attorney", r"counsel", r"lawyer", r"solicitor", r"advocate", r"representing", r"esq\.?"],
            "witness": [r"witness", r"testified", r"deposed", r"affiant"]
        }
        
        try:
            # Check if NER model already loaded
            if hasattr(EntityAnonymizer, 'shared_ner_pipeline'):
                self.ner_pipeline = EntityAnonymizer.shared_ner_pipeline
                logger.info("Using already loaded NER model")
            else:
                from transformers import pipeline
                # Initialize NER pipeline for entity detection
                logger.info("Loading NER model for entity anonymization...")
                self.ner_pipeline = pipeline(
                    "ner",
                    model="jean-baptiste/roberta-large-ner-english",
                    tokenizer="jean-baptiste/roberta-large-ner-english",
                    aggregation_strategy="simple",
                    device=0 if device == "cuda" else -1
                )
                # Save as class attribute for reuse
                EntityAnonymizer.shared_ner_pipeline = self.ner_pipeline
                logger.info("Initialized NER model for entity anonymization")
        except Exception as e:
            logger.warning(f"Failed to initialize NER model, falling back to rule-based anonymization: {e}")
            self.ner_pipeline = None
    
    def identify_entities(self, text: str) -> None:
        """Identify entities in the text and categorize them by role."""
        # First pass: Use NER to identify potential entities
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                for entity in entities:
                    if entity["entity_group"] in ["PER", "B-PER", "I-PER"]:
                        self._categorize_entity(entity["word"], text)
                    elif entity["entity_group"] in ["ORG", "B-ORG", "I-ORG"]:
                        if entity["word"] not in self.entity_map:
                            self.entity_map[entity["word"]] = f"Organization_{len(self.entity_types['organization'])+1}"
                            self.entity_types["organization"].append(entity["word"])
            except Exception as e:
                logger.warning(f"NER processing failed: {e}. Using rule-based approach.")
        
        # Second pass: Rule-based identification for legal roles
        self._identify_legal_roles(text)
            
    def _categorize_entity(self, entity: str, context: str) -> None:
        """Categorize an entity based on its context in the text."""
        if entity in self.entity_map:
            return
            
        # Check surrounding context to determine role
        window_size = 100  # Check 100 chars before and after entity mentions
        
        # Find all instances of this entity in the text
        for match in re.finditer(re.escape(entity), context):
            start = max(0, match.start() - window_size)
            end = min(len(context), match.end() + window_size)
            surrounding_text = context[start:end].lower()
            
            # Check for role patterns in the surrounding text
            role_assigned = False
            for role, patterns in self.legal_role_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, surrounding_text):
                        if entity not in self.entity_types[role]:
                            self.entity_types[role].append(entity)
                            role_assigned = True
                            break
                if role_assigned:
                    break
            
            # If no specific role identified, categorize as 'other'
            if not role_assigned and entity not in self.entity_types["other"]:
                self.entity_types["other"].append(entity)
                
        # Generate anonymized name based on role and order
        for role, entities in self.entity_types.items():
            if entity in entities:
                idx = entities.index(entity) + 1
                if role == "plaintiff":
                    self.entity_map[entity] = f"Plaintiff_{idx}" if idx > 1 else "Plaintiff"
                elif role == "defendant":
                    self.entity_map[entity] = f"Defendant_{idx}" if idx > 1 else "Defendant"
                elif role == "judge":
                    self.entity_map[entity] = f"Judge_{idx}" if idx > 1 else "Judge"
                elif role == "attorney":
                    self.entity_map[entity] = f"Attorney_{idx}" if idx > 1 else "Attorney"
                elif role == "witness":
                    self.entity_map[entity] = f"Witness_{idx}" if idx > 1 else "Witness"
                elif role == "organization":
                    self.entity_map[entity] = f"Organization_{idx}"
                else:
                    self.entity_map[entity] = f"Entity_{idx}"
                break
    
    def _identify_legal_roles(self, text: str) -> None:
        """Identify common legal role patterns in document."""
        # Common patterns like "John Doe, Plaintiff" or "ABC Corp (Defendant)"
        patterns = [
            # NAME, ROLE pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s+(plaintiff|defendant|petitioner|respondent)',
            # NAME (ROLE) pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+\((plaintiff|defendant|petitioner|respondent)\)',
            # Organization patterns
            r'([A-Z][A-Za-z0-9\s&,.]+(?:Inc\.|Corp\.|LLC|Ltd\.|Limited|Company))',
            # v. pattern (common in case names)
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    name, role = match.groups()
                    if "plaintiff" in role.lower() or "petitioner" in role.lower():
                        if name not in self.entity_types["plaintiff"]:
                            self.entity_types["plaintiff"].append(name)
                    elif "defendant" in role.lower() or "respondent" in role.lower():
                        if name not in self.entity_types["defendant"]:
                            self.entity_types["defendant"].append(name)
                else:
                    # For org patterns or other single-capture patterns
                    entity = match.group(1)
                    if entity not in self.entity_map:
                        if re.search(r'Inc\.|Corp\.|LLC|Ltd\.|Limited|Company', entity):
                            self.entity_types["organization"].append(entity)
                        else:
                            self.entity_types["other"].append(entity)
                            
        # Process "v." pattern separately to identify opposing parties
        v_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        for match in re.finditer(v_pattern, text):
            if len(match.groups()) == 2:
                plaintiff, defendant = match.groups()
                if plaintiff not in self.entity_types["plaintiff"]:
                    self.entity_types["plaintiff"].append(plaintiff)
                if defendant not in self.entity_types["defendant"]:
                    self.entity_types["defendant"].append(defendant)
        
        # Update entity map for all identified entities
        for role, entities in self.entity_types.items():
            for i, entity in enumerate(entities):
                if entity not in self.entity_map:
                    idx = i + 1
                    if role == "plaintiff":
                        self.entity_map[entity] = f"Plaintiff_{idx}" if idx > 1 else "Plaintiff"
                    elif role == "defendant":
                        self.entity_map[entity] = f"Defendant_{idx}" if idx > 1 else "Defendant"
                    elif role == "judge":
                        self.entity_map[entity] = f"Judge_{idx}" if idx > 1 else "Judge"
                    elif role == "attorney":
                        self.entity_map[entity] = f"Attorney_{idx}" if idx > 1 else "Attorney"
                    elif role == "witness":
                        self.entity_map[entity] = f"Witness_{idx}" if idx > 1 else "Witness"
                    elif role == "organization":
                        self.entity_map[entity] = f"Organization_{idx}"
                    else:
                        self.entity_map[entity] = f"Entity_{idx}"
    
    def anonymize_text(self, text: str) -> str:
        """Replace identified entities with anonymous placeholders."""
        # First identify entities in the text if not already done
        if not self.entity_map:
            self.identify_entities(text)
            
        # Replace entities with their anonymized versions
        anonymized_text = text
        
        # Sort entities by length (descending) to avoid partial replacements
        sorted_entities = sorted(self.entity_map.keys(), key=len, reverse=True)
        
        for entity in sorted_entities:
            anonymized_text = re.sub(
                r'\b' + re.escape(entity) + r'\b', 
                self.entity_map[entity], 
                anonymized_text
            )
            
        return anonymized_text
    
    def get_entity_map(self) -> Dict[str, str]:
        """Return the mapping of original entities to anonymized versions."""
        return self.entity_map.copy()

class PDFProcessor:
    """Processor for PDF documents."""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize the PDF processor.
        
        Args:
            model_manager: ModelManager instance for text processing
        """
        self.model_manager = model_manager
        self.temp_dir = tempfile.mkdtemp()
        self.anonymizer = EntityAnonymizer(device="cuda" if torch.cuda.is_available() else "cpu")
    
    def __del__(self):
        """Clean up temporary files on deletion."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def extract_text_and_images(self, pdf_path: str) -> Tuple[str, List[Image.Image]]:
        """
        Extract text and images from a PDF file.
        
        Args:
            pdf_path: Path or Drive reference to the PDF file
            
        Returns:
            Tuple containing the extracted text and a list of page images
        """
        try:
            # Handle Google Drive paths
            if pdf_path.startswith("gdrive://"):
                file_id = pdf_path.split('/')[2]
                file_name = pdf_path.split('/')[-1]
                file_info = {'id': file_id, 'name': file_name}
                local_path = GoogleDriveHandler.download_pdf(file_info, self.temp_dir)
                if not local_path:
                    return "", []
                pdf_path = local_path
            
            # Extract text using PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            # Extract images using pdf2image
            images = convert_from_path(pdf_path, dpi=150)
            
            return text, images
        except Exception as e:
            logger.error(f"Error extracting content from PDF {pdf_path}: {str(e)}")
            traceback.print_exc()
            return "", []

    def process_pdf(self, pdf_path: str) -> Optional[CaseEntry]:
        """
        Process a PDF file and extract structured information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            CaseEntry if processing succeeded, None otherwise
        """
        try:
            # Extract case ID from filename
            filename = os.path.basename(pdf_path)
            case_id = f"CONTRACT_{filename.split('.')[0]}"
            if not case_id.startswith("CONTRACT_"):
                case_id = f"CONTRACT_{case_id}"
            
            # Extract text and images
            text, images = self.extract_text_and_images(pdf_path)
            if not text or not images:
                logger.warning(f"Failed to extract content from {pdf_path}")
                return None
            
            # Anonymize text to protect privacy
            logger.info(f"Anonymizing entities in document {case_id}")
            anonymized_text = self.anonymizer.anonymize_text(text)
            entity_map = self.anonymizer.get_entity_map()
            logger.info(f"Identified and anonymized {len(entity_map)} entities")
            
            # Extract layout information (sections, etc.)
            layout_info = self.model_manager.extract_layout_info(images[0], anonymized_text)
            
            # Generate scenario summary
            scenario = self.model_manager.summarize_with_t5(
                anonymized_text, 
                "Summarize this contract case scenario:", 
                max_length=200
            )
            
            # Extract questions
            questions = self.model_manager.extract_questions(anonymized_text)
            
            # Extract relevant sections
            relevant_sections = layout_info.get("sections", [])
            
            # Generate suggested actions
            suggested_actions = self.model_manager.generate_suggested_actions(anonymized_text)
            
            # Generate strategic advice
            strategic_advice = self.model_manager.generate_strategic_advice(anonymized_text)
            
            # Generate estimated outcome
            estimated_outcome = self.model_manager.summarize_with_t5(
                anonymized_text,
                "Provide a brief summary of the likely outcome for this contract case:",
                max_length=100
            )
            
            # Generate embedding for the scenario
            scenario_embedding = self.model_manager.generate_embedding(scenario)
            
            # Create and return case entry
            return CaseEntry(
                case_id=case_id,
                scenario=scenario,
                questions=questions,
                relevant_sections=relevant_sections,
                suggested_actions=suggested_actions,
                strategic_advice=strategic_advice,
                estimated_outcome=estimated_outcome,
                scenario_embedding=scenario_embedding
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            traceback.print_exc()
            return None

class DatasetGenerator:
    """Generator for the contract cases dataset."""
    
    def __init__(
        self, 
        pdf_folder: str, 
        output_path: str,
        num_workers: int,
        use_gpu: bool
    ):
        """
        Initialize the dataset generator.
        
        Args:
            pdf_folder: Folder containing PDF files
            output_path: Path for the output JSON file
            num_workers: Number of parallel workers
            use_gpu: Whether to use GPU
        """
        self.pdf_folder = pdf_folder
        self.output_path = output_path
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        
        # Initialize components
        self.model_manager = ModelManager(use_gpu)
        self.pdf_processor = PDFProcessor(self.model_manager)
        
    def process_batch(
        self, 
        pdf_files: List[Dict[str, str]], 
        start_idx: int, 
        batch_size: int
    ) -> List[CaseEntry]:
        """
        Process a batch of PDF files.
        
        Args:
            pdf_files: List of PDF file info dictionaries
            start_idx: Start index in the file list
            batch_size: Number of files to process
            
        Returns:
            List of processed CaseEntry objects
        """
        results = []
        end_idx = min(start_idx + batch_size, len(pdf_files))
        batch = pdf_files[start_idx:end_idx]
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor when using GPU models
        # This prevents issues with CUDA contexts between processes
        executor_class = concurrent.futures.ThreadPoolExecutor if self.use_gpu else concurrent.futures.ProcessPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            # Pass the path from the file info
            future_to_pdf = {
                executor.submit(self.pdf_processor.process_pdf, pdf_info['path']): pdf_info 
                for pdf_info in batch
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_pdf), 
                              total=len(batch), 
                              desc=f"Processing batch {start_idx//batch_size + 1}"):
                pdf_info = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Exception processing {pdf_info['name']}: {str(e)}")
                    traceback.print_exc()
        
        return results
    
    def save_results(self, results: List[CaseEntry], is_final: bool = False):
        """
        Save results to JSON file.
        
        Args:
            results: List of CaseEntry objects to save
            is_final: Whether this is the final save
        """
        # Convert dataclass objects to dictionaries
        results_dict = [asdict(entry) for entry in results]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to file
        output_file = self.output_path
        if not is_final:
            # For intermediate saves, add timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{os.path.splitext(self.output_path)[0]}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved {len(results)} entries to {output_file}")
    
    def generate_dataset(self, batch_size: int = 50, save_interval: int = 500):
        """
        Generate the dataset from PDF files.
        
        Args:
            batch_size: Number of PDFs to process in each batch
            save_interval: Number of processed entries after which to save intermediate results
        """
        # Scan for PDF files
        pdf_files = GoogleDriveHandler.scan_for_pdfs(self.pdf_folder)
        if not pdf_files:
            logger.error(f"No PDF files found in {self.pdf_folder}")
            return
        
        # Process PDFs in batches
        all_results = []
        for start_idx in range(0, len(pdf_files), batch_size):
            batch_results = self.process_batch(pdf_files, start_idx, batch_size)
            
            # Add to total results
            all_results.extend(batch_results)
            
            # Save intermediate results
            if len(all_results) % save_interval < batch_size:
                self.save_results(all_results, is_final=False)
            
            # Clear GPU memory
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            # Garbage collection
            gc.collect()
        
        # Save final results
        self.save_results(all_results, is_final=True)
        logger.info(f"Dataset generation complete. Total entries: {len(all_results)}")

def main():
    """Main function."""
    # Add installation for needed NER package
    logger.info("Installing additional required packages for entity anonymization...")
    try:
        !pip install spacy
    except:
        logger.warning("Failed to install spacy via !pip command, you may need to install it manually")
    
    # Mount Google Drive (if running in Colab)
    drive_mounted = GoogleDriveHandler.mount_drive()
    
    # Detect hardware
    hardware_info = HardwareManager.detect_hardware()
    num_workers = HardwareManager.calculate_optimal_workers(hardware_info, max_resource_usage=0.95)
    use_gpu = hardware_info["gpu_available"]
    
    # Define paths
    default_folder_name = "law_test"  # Target folder name
    default_output_path = "output/contract_cases_dataset.json"  # Output folder name
    
    # Allow user to input custom paths
    custom_pdf_path = input(f"Enter name of Google Drive folder containing PDFs (default: {default_folder_name}): ").strip()
    pdf_folder = custom_pdf_path if custom_pdf_path else default_folder_name
    
    # Handle both relative and absolute paths
    if not pdf_folder.startswith('/'):
        # If it's a relative path, make it relative to mounted drive
        if drive_mounted:
            pdf_folder = f"/content/drive/MyDrive/{pdf_folder}"
    
    custom_output_path = input(f"Enter output JSON path (default: {default_output_path}): ").strip()
    output_path = custom_output_path if custom_output_path else default_output_path
    
    # Handle both relative and absolute paths for output
    if not output_path.startswith('/'):
        # If it's a relative path, make it relative to mounted drive
        if drive_mounted:
            output_path = f"/content/drive/MyDrive/{output_path}"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path is not just a filename
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate dataset
    generator = DatasetGenerator(
        pdf_folder=pdf_folder,
        output_path=output_path,
        num_workers=num_workers,
        use_gpu=use_gpu
    )
    
    # Start generation
    logger.info(f"Starting dataset generation from folder '{pdf_folder}'")
    generator.generate_dataset(batch_size=5, save_interval=20)  # Smaller batches for Drive files
    logger.info("Dataset generation complete")

if __name__ == "__main__":
    main()
