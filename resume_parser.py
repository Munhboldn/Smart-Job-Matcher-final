import fitz  # PyMuPDF
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
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with improved formatting preservation."""
    text = ""
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text += block[4] + "\n\n"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
    return clean_resume_text(text)

def extract_text_from_docx(docx_file):
    """Extract text from DOCX with improved structure preservation."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(docx_file.read())
            tmp_path = tmp.name
        doc = Document(tmp_path)
        full_text = []
        for header in doc.sections[0].header.paragraphs:
            if header.text.strip():
                full_text.append(header.text.strip())
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    full_text.append(" | ".join(row_text))
        os.unlink(tmp_path)
        return clean_resume_text("\n\n".join(full_text))
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def clean_resume_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('•', '\n• ')
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

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
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line_lower) and len(line_lower) < 50:
                current_section = section_name
                break
        if i < len(lines) - 1:
            sections[current_section] += lines[i+1] + "\n"
    if len(sections) <= 1:
        sections['content'] = text
    return dict(sections)

def extract_contact_info(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\+?[\d\s()-]{7,})'
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    linkedin = re.findall(linkedin_pattern, text)
    doc = nlp(text[:1000])
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return {
        'emails': emails,
        'phones': phones,
        'linkedin': linkedin,
        'locations': locations
    }

def analyze_resume(text):
    sections = extract_resume_sections(text)
    contact_info = extract_contact_info(text)
    doc = nlp(text[:5000])
    entities = {
        'organizations': [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        'dates': [ent.text for ent in doc.ents if ent.label_ == "DATE"],
        'people': [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    }
    section_weights = {
        'contact': 10,
        'education': 20,
        'experience': 30,
        'skills': 20,
        'projects': 10,
        'achievements': 5,
        'languages': 5
    }
    completeness_score = 0
    for section, weight in section_weights.items():
        if section in sections and len(sections[section]) > 10:
            completeness_score += weight
    if contact_info['emails']:
        completeness_score += 5
    if contact_info['phones']:
        completeness_score += 5
    return {
        'sections': sections,
        'contact_info': contact_info,
        'entities': entities,
        'completeness_score': completeness_score
    }
