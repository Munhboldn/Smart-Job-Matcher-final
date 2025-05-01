import re
import os
import spacy
import docx2txt
import fitz # PyMuPDF
from docx import Document
from io import BytesIO
import nltk
from nltk.corpus import stopwords
import logging
import streamlit as st # Import streamlit for caching

logger = logging.getLogger(__name__) # Get the logger configured in app.py

# --- NLTK Data Downloads ---
# Ensure NLTK data is available. Downloads if not found.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# --- SpaCy Model Loading ---
# Load spaCy model (v3+ compatible)
# Use @st.cache_resource to load the model only once across Streamlit reruns
@st.cache_resource
def load_spacy_model(model_name):
    """Loads a spaCy model and caches it."""
    try:
        logger.info(f"Attempting to load spaCy model '{model_name}' for basic parsing.")
        nlp_model = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model '{model_name}' for basic parsing.")
        return nlp_model
    except OSError as e:
        error_msg = f"FATAL ERROR: Could not load spaCy model '{model_name}'. Ensure it is correctly listed and installed via requirements.txt."
        logger.error(error_msg, exc_info=True)
        raise # Re-raise if model loading is critical

nlp = load_spacy_model("en_core_web_sm")


# --- Text Extraction Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file-like object using PyMuPDF."""
    logger.info("Extracting text from PDF")
    text = ""
    try:
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
        logger.info("PDF extraction successful.")
    except Exception as e:
        logger.error(f"PDF extraction error: {e}", exc_info=True)
        text = "Error extracting PDF content."
    return text

def extract_text_from_docx(docx_file):
    """Extracts text from a DOCX file-like object using python-docx, with docx2txt fallback."""
    logger.info("Extracting text from DOCX")
    text = ""
    try:
        docx_file.seek(0)
        document = Document(BytesIO(docx_file.getvalue()))
        paragraphs = [para.text for para in document.paragraphs]
        text = '\n'.join(paragraphs)
        logger.info("DOCX extraction using python-docx successful.")

        # Fallback if python-docx extracts too little text (e.g., scanned doc)
        if len(text.strip()) < 100:
            logger.info("Text too short from python-docx, trying docx2txt fallback.")
            docx_file.seek(0) # Reset file pointer for docx2txt
            text = docx2txt.process(docx_file)
            logger.info("DOCX extraction using docx2txt successful.")

    except Exception as e:
        logger.error(f"DOCX extraction error: {e}", exc_info=True)
        text = "Error extracting DOCX content."
    return text

# --- Resume Parsing Functions ---

def extract_resume_sections(text):
    """Attempts to extract sections from resume text based on common headers."""
    # This is a heuristic method and may need tuning based on resume formats
    section_headers = [
        "Education", "Experience", "Work Experience", "Employment", "Skills",
        "Technical Skills", "Projects", "Certifications", "Awards", "Publications",
        "Languages", "Interests", "Objective", "Summary", "Profile", "Contact",
        "References", "Achievements", "Professional Experience", "Academic Projects"
    ]
    sections = {}
    current_section = "Header"
    section_content = []
    lines = text.split('\n')

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Use regex to find full header matches (case-insensitive, handles plural/colon)
        matched_header = next((h for h in section_headers if re.fullmatch(f"{re.escape(h)}s?\\s*:?", line_clean, re.IGNORECASE)), None)

        if matched_header:
            if section_content:
                sections[current_section] = "\n".join(section_content).strip()

            current_section = matched_header
            section_content = []
        else:
            section_content.append(line_clean)

    if section_content:
        sections[current_section] = "\n".join(section_content).strip()

    return sections

def extract_skills(text):
    """Extracts skills based on a predefined list and spaCy noun chunks."""
    # Common skills list (can be expanded)
    common_skills = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "PHP", "Go", "Rust",
        "SQL", "NoSQL", "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
        "TensorFlow", "PyTorch", "Keras", "Pandas", "NumPy", "scikit-learn", "AWS", "Azure",
        "GCP", "Docker", "Kubernetes", "CI/CD", "Git", "Jira", "Leadership", "Communication",
        "Teamwork", "Problem Solving", "Machine Learning", "Artificial Intelligence", "AI", "ML",
        "Data Science", "Cloud Computing", "Big Data", "Agile", "Scrum"
    ]
    found_skills = set()

    # Use regex to find exact or close matches from the list (case-insensitive, word boundaries)
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(skill) for skill in common_skills) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    found_skills.update(matches)

    # Use spaCy to find noun chunks (potential phrases) and check relevance
    doc = nlp(text) # Use the cached nlp model
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()

        # Add noun chunks that match a common skill exactly (case-insensitive)
        if chunk_lower in (skill.lower() for skill in common_skills):
             found_skills.add(chunk_text)

        # Add multi-word noun chunks that contain a common skill (heuristic)
        elif len(chunk_text.split()) > 1 and any(skill.lower() in chunk_lower for skill in common_skills):
             # Simple check to avoid adding single words already found
             if chunk_lower not in (skill.lower() for skill in common_skills):
                  found_skills.add(chunk_text)

    return list(found_skills)


def analyze_resume(text):
    """Analyzes the resume text using basic section parsing and skill extraction."""
    logger.info("Starting basic resume analysis")
    result = {
        "name": "",
        "email": "",
        "phone": "",
        "education": [],
        "experience": [],
        "skills": []
    }

    sections = extract_resume_sections(text)

    # --- Extract Contact Info (Basic) ---
    email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_matches:
        result["email"] = email_matches[0]

    phone_matches = re.findall(
        r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}(?:\s*(?:ext|x|extn|extension)\s*\d+)?|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{4})',
        text
    )
    if phone_matches:
         result["phone"] = ''.join(phone_matches[0]) if isinstance(phone_matches[0], tuple) else phone_matches[0]

    # --- Extract Name (Heuristic) ---
    if "Header" in sections and sections['Header']:
        header_lines = [line.strip() for line in sections["Header"].split('\n') if line.strip()]
        if header_lines:
            result["name"] = header_lines[0]

    # --- Extract Education ---
    if "Education" in sections:
        result["education"] = [e.strip() for e in sections["Education"].split('\n') if len(e.strip()) > 5]

    # --- Extract Experience ---
    for key in ["Experience", "Work Experience", "Employment", "Professional Experience"]:
        if key in sections:
            entries = [e.strip() for e in sections[key].split('\n') if len(e.strip()) > 10]
            result["experience"].extend(entries)
            break

    # --- Extract Skills ---
    result["skills"] = extract_skills(text)

    logger.info("Basic resume analysis complete.")
    return result

# Expose nlp for debug check in app.py if needed
nlp_model_for_debug = nlp

# Keep other functions like extract_contact_info or other analyze_resume versions
# only if they are intended to be used by functions *within* this module.
# Since the old app.py imports analyze_resume and extract_resume_sections from here,
# and analyze_resume calls extract_resume_sections, those are kept.
# The old app.py also imported analyze_resume and extract_skills_from_resume from semantic_matcher.
# It also had different text extraction and a different analyze_resume in the second old file.
# To match the old app.py's imports, we'll rely on functions being in the module app.py expects.
# Based on the old app.py imports:
# - extract_text_from_pdf, extract_text_from_docx, analyze_resume, extract_resume_sections are needed here.
# - semantic_match_resume, extract_resume_keywords, extract_skills_from_resume, get_skill_matches, analyze_resume (aliased) are needed in semantic_matcher.
# Let's rename extract_skills to extract_skills_basic to avoid confusion with the semantic version.
# And ensure analyze_resume is the basic section one as imported by the old app.py.

# Let's rename functions here to match the old app.py's expected imports if necessary
# Old app.py imports: extract_text_from_pdf, extract_text_from_docx, analyze_resume, extract_resume_sections
# Current functions: extract_text_from_pdf, extract_text_from_docx, analyze_resume, extract_resume_sections, extract_skills
# This seems consistent, except for extract_skills. Let's rename it.

def extract_skills_basic(text): # Renamed for clarity
    """Extracts skills based on a predefined list and spaCy noun chunks (Basic method)."""
    common_skills = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "PHP", "Go", "Rust",
        "SQL", "NoSQL", "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
        "TensorFlow", "PyTorch", "Keras", "Pandas", "NumPy", "scikit-learn", "AWS", "Azure",
        "GCP", "Docker", "Kubernetes", "CI/CD", "Git", "Jira", "Leadership", "Communication",
        "Teamwork", "Problem Solving", "Machine Learning", "Artificial Intelligence", "AI", "ML",
        "Data Science", "Cloud Computing", "Big Data", "Agile", "Scrum"
    ]
    found_skills = set()
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(skill) for skill in common_skills) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    found_skills.update(matches)
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()
        if chunk_lower in (skill.lower() for skill in common_skills):
             found_skills.add(chunk_text)
        elif len(chunk_text.split()) > 1 and any(skill.lower() in chunk_lower for skill in common_skills):
             if chunk_lower not in (skill.lower() for skill in common_skills):
                  found_skills.add(chunk_text)
    return list(found_skills)

# Update analyze_resume to call the renamed extract_skills_basic
def analyze_resume(text):
    """Analyzes the resume text using basic section parsing and skill extraction."""
    logger.info("Starting basic resume analysis")
    result = {
        "name": "", "email": "", "phone": "",
        "education": [], "experience": [], "skills": []
    }
    sections = extract_resume_sections(text)
    email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_matches: result["email"] = email_matches[0]
    phone_matches = re.findall(r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}(?:\s*(?:ext|x|extn|extension)\s*\d+)?|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{4})', text)
    if phone_matches: result["phone"] = ''.join(phone_matches[0]) if isinstance(phone_matches[0], tuple) else phone_matches[0]
    if "Header" in sections and sections['Header']:
        header_lines = [line.strip() for line in sections["Header"].split('\n') if line.strip()]
        if header_lines: result["name"] = header_lines[0]
    if "Education" in sections:
        result["education"] = [e.strip() for e in sections["Education"].split('\n') if len(e.strip()) > 5]
    for key in ["Experience", "Work Experience", "Employment", "Professional Experience"]:
        if key in sections:
            entries = [e.strip() for e in sections[key].split('\n') if len(e.strip()) > 10]
            result["experience"].extend(entries)
            break
    result["skills"] = extract_skills_basic(text) # Call the renamed function
    logger.info("Basic resume analysis complete.")
    return result
