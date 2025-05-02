import fitz # PyMuPDF
import docx2txt
import tempfile
import re
import os
from docx import Document
import pandas as pd
import spacy
from collections import defaultdict

# Load spaCy model for NER
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # Fallback for environments where direct download might be restricted
    # In a Streamlit environment, ensure this package is in your requirements.txt
    # or pre-installed. This subprocess call might not work everywhere.
    try:
        import subprocess
        # Use --user to install in user site-packages if necessary
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load('en_core_web_sm')
        print("spaCy model downloaded and loaded successfully.")
    except Exception as e:
        print(f"Error loading or downloading spaCy model: {e}")
        print("Proceeding without spaCy NER features.")
        nlp = None # Set nlp to None if loading fails

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with improved formatting preservation."""
    text = ""
    try:
        # Move file pointer to the beginning in case it was read before
        pdf_file.seek(0)
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                # Use 'text' with 'textenabled=True' for potentially better block order
                # Or stick to 'blocks' but process them carefully
                # Let's refine the 'blocks' processing slightly
                blocks = page.get_text("blocks")
                # Sort blocks by their top-left corner (y then x) to handle columns
                blocks.sort(key=lambda block: (block[1], block[0]))
                for block in blocks:
                     block_text = block[4].strip()
                     if block_text:
                          text += block_text + "\n\n" # Add extra newline between blocks
    except Exception as e:
        # Log the error or handle it appropriately
        print(f"Error extracting PDF: {str(e)}")
        return f"Error extracting PDF: {str(e)}" # Return error message to caller
    return clean_resume_text(text)

def extract_text_from_docx(docx_file):
    """Extract text from DOCX with improved structure preservation."""
    try:
        # Move file pointer to the beginning
        docx_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(docx_file.read())
            tmp_path = tmp.name

        doc = Document(tmp_path)
        full_text = []

        # Extract header text
        for section in doc.sections:
             for header in section.header.paragraphs:
                  if header.text.strip():
                       full_text.append(header.text.strip())

        # Extract main body paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        # Extract table text
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    full_text.append(" | ".join(row_text)) # Join cell text with a separator

        os.unlink(tmp_path) # Clean up the temporary file
        return clean_resume_text("\n\n".join(full_text)) # Join parts with double newline

    except Exception as e:
        # Log the error or handle it appropriately
        print(f"Error extracting DOCX: {str(e)}")
        return f"Error extracting DOCX: {str(e)}" # Return error message to caller


def clean_resume_text(text):
    """Basic text cleaning for resume content."""
    # Remove excessive whitespace, keeping some structure
    text = re.sub(r'[ \t]+', ' ', text) # Replace multiple spaces/tabs with a single space
    text = re.sub(r'\n\s*\n', '\n\n', text) # Replace multiple newlines with double newline
    text = text.replace('•', '\n• ') # Ensure bullet points are on new lines
    # Add a space between a lowercase letter and a following uppercase letter (e.g., "skillSet" -> "skill Set")
    # This can sometimes help split concatenated words, but be cautious as it might split valid words (e.g., "iPhone")
    # Let's make this optional or more targeted if needed. For now, keep it as it was.
    # text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) # Keep this as per original code
    text = re.sub(r'\n{3,}', '\n\n', text) # Reduce excessive newlines

    return text.strip()


def extract_resume_sections(text):
    """
    Extracts content into standard resume sections based on common headers.
    Improved logic to collect content until the next header.
    """
    # More robust patterns including common variations and allowing for case-insensitivity
    section_patterns = {
        'contact': r'^\s*(contact|personal|info|information|profile|details)\s*$',
        'summary': r'^\s*(summary|objective|about)\s*$', # Added Summary/Objective
        'education': r'^\s*(education|academic|qualification|degree|university|school)\s*$',
        'experience': r'^\s*(experience|employment|work|history|professional|career)\s*$',
        'skills': r'^\s*(skills|abilities|expertise|c
