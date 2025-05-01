import re
import os
import spacy
import docx2txt
import fitz # PyMuPDF
from docx import Document
from io import BytesIO
import nltk
# Removed CountVectorizer as it wasn't used in the final analyze_resume logic
from nltk.corpus import stopwords
import logging
# Removed `subprocess` import as it's no longer needed or correct here

# Configure logging
# Basic config might be set up in app.py, but harmless to have here too
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
# This now relies SOLELY on the model being installed by requirements.txt
try:
    logger.info("Attempting to load spaCy model 'en_core_web_sm'")
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model 'en_core_web_sm'")
except OSError as e:
    error_msg = f"FATAL ERROR: Could not load spaCy model 'en_core_web_sm'. " \
                f"Please ensure it is correctly listed and installed via your requirements.txt file. " \
                f"Full error details: {e}"
    logger.error(error_msg, exc_info=True)
    raise


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

        if len(text.strip()) < 100:
            logger.info("Text too short from python-docx, trying docx2txt fallback.")
            docx_file.seek(0)
            text = docx2txt.process(docx_file)
            logger.info("DOCX extraction using docx2txt successful.")

    except Exception as e:
        logger.error(f"DOCX extraction error: {e}", exc_info=True)
        text = "Error extracting DOCX content."
    return text

# --- Resume Parsing Functions ---

def extract_resume_sections(text):
    """Attempts to extract sections from resume text based on common headers."""
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

    if 'Header' in sections and sections['Header']:
         first_matched_section_key = next((h for h in section_headers if h in sections and h != 'Header'), None)
         if first_matched_section_key:
             header_lines = sections['Header'].split('\n')
             temp_header_content = []
             moved_lines = []
             is_in_header = True
             for line in header_lines:
                  line_clean = line.strip()
                  if re.search(r'^\s*•\s+|\d{4}|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|\b(present|current)\b', line_clean, re.IGNORECASE):
                       is_in_header = False

                  if is_in_header:
                       temp_header_content.append(line)
                  else:
                       moved_lines.append(line)

             if moved_lines:
                  sections['Header'] = "\n".join(temp_header_content).strip()
                  original_first_section_content = sections.get(first_matched_section_key, "")
                  sections[first_matched_section_key] = "\n".join(moved_lines).strip() + "\n" + original_first_section_content.strip()


    return sections

def extract_skills(text):
    """Extracts skills based on a predefined list and spaCy noun chunks."""
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

    # Use spaCy to find noun chunks
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()

        if chunk_lower in (skill.lower() for skill in common_skills):
             found_skills.add(chunk_text)

        elif any(skill.lower() in chunk_lower for skill in common_skills) and chunk_lower not in (skill.lower() for skill in common_skills):
             if len(chunk_text.split()) > 1 and len(chunk_text) > len(max((s for s in common_skills if s.lower() in chunk_lower), key=len, default="")):
                  found_skills.add(chunk_text)

    return list(found_skills)


def analyze_resume(text):
    """Analyzes the resume text to extract key information like contact, education, experience, and skills."""
    logger.info("Analyzing resume (basic parser)")
    result = {
        "name": "",
        "email": "",
        "phone": "",
        "education": [],
        "experience": [],
        "skills": []
    }

    sections = extract_resume_sections(text)

    # --- Extract Contact Info ---
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

    return result
