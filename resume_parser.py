import re
import os
import spacy
import fitz  # PyMuPDF
import docx2txt
import tempfile
from docx import Document
from collections import defaultdict
import logging
import streamlit as st

# --- Logging ---
logger = logging.getLogger(__name__)


@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        logger.info(f"Attempting to load spaCy model '{model_name}' for basic parsing.")
        nlp_model = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model '{model_name}' for basic parsing.")
        return nlp_model
    except OSError as e:
        error_msg = f"FATAL ERROR: Could not load spaCy model '{model_name}'. Ensure it is correctly listed and installed via requirements.txt."
        logger.error(error_msg, exc_info=True)
        raise

nlp = load_spacy_model()

# --- Text Extraction Functions ---
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF preserving formatting."""
    text = ""
    try:
        pdf_file.seek(0)
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text += block[4] + "\n\n"
        return clean_resume_text(text)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return "Error extracting PDF content."

def extract_text_from_docx(docx_file):
    """Extract structured text from DOCX files."""
    try:
        docx_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(docx_file.read())
            tmp_path = tmp.name

        doc = Document(tmp_path)
        os.unlink(tmp_path)

        full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return clean_resume_text("\n\n".join(full_text))

    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return "Error extracting DOCX content."

# --- Text Cleaning Function ---
def clean_resume_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('•', '\n• ')
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()

# --- Resume Section Extraction ---
def extract_resume_sections(text):
    section_patterns = {
        'contact': r'(contact|personal|info|information|profile|details)',
        'education': r'(education|academic|qualification|degree|university|school)',
        'experience': r'(experience|employment|work|history|professional)',
        'skills': r'(skills|abilities|expertise|competencies|proficiencies)',
        'projects': r'(projects|portfolio|works)',
        'achievements': r'(achievements|accomplishments|honors|awards)',
        'languages': r'(languages|linguistic)',
        'references': r'(references|recommendations)'
    }
    sections = defaultdict(str)
    lines = text.split('\n')
    current_section = 'other'
    for line in lines:
        line_lower = line.lower().strip()
        matched = False
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line_lower) and len(line_lower) < 50:
                current_section = section_name
                matched = True
                break
        if not matched:
            sections[current_section] += line + "\n"
    return dict(sections)

# --- Skill Extraction Enhanced ---
def extract_skills(text):
    common_skills = ["Python", "Java", "SQL", "JavaScript", "Machine Learning", "Docker", "AWS", "React", "Git"]
    found_skills = set()
    pattern = re.compile(r'\\b(?:' + '|'.join(re.escape(skill) for skill in common_skills) + r')\\b', re.I)
    found_skills.update(pattern.findall(text))
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if any(skill.lower() in chunk_text.lower() for skill in common_skills):
            found_skills.add(chunk_text)
    return list(found_skills)

# --- Contact Information Extraction ---
def extract_contact_info(text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    phone = re.findall(r'(\+?[\d\s()-]{7,})', text)
    linkedin = re.findall(r'linkedin\.com/in/[\w-]+', text)
    locations = [ent.text for ent in nlp(text).ents if ent.label_ == "GPE"]
    return {'email': email, 'phone': phone, 'linkedin': linkedin, 'location': locations}

# --- Resume Completeness Scoring ---
def analyze_resume(text):
    sections = extract_resume_sections(text)
    contact_info = extract_contact_info(text)
    completeness_weights = {
        'contact': 10, 'education': 20, 'experience': 30,
        'skills': 20, 'projects': 10, 'achievements': 5,
        'languages': 5
    }
    completeness_score = sum(weight for sec, weight in completeness_weights.items() if sections.get(sec))
    completeness_score += 5 if contact_info.get('email') else 0
    completeness_score += 5 if contact_info.get('phone') else 0

    return {
        'sections': sections,
        'contact_info': contact_info,
        'skills': extract_skills(text),
        'completeness_score': completeness_score
    }
