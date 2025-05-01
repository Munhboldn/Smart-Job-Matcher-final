import fitz  # PyMuPDF
import docx2txt
import re
import os
import tempfile
from docx import Document
from collections import Counter, defaultdict
from typing import Dict, List, Any
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return clean_text(text)


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file using docx2txt."""
    return clean_text(docx2txt.process(docx_path))


def clean_text(text: str) -> str:
    """Clean and normalize resume text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('•', '\n•')
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()


def detect_resume_sections(text: str) -> Dict[str, bool]:
    section_keywords = {
        "Education": r"(Education|Боловсрол|сургалт|их сургууль)",
        "Experience": r"(Experience|Туршлага|ажилласан|ажлын туршлага)",
        "Skills": r"(Skills|Ур чадвар|чадварууд)",
        "Projects": r"(Projects|Төслүүд|portfolio)",
        "Certifications": r"(Certifications|Гэрчилгээ|сертификат)",
        "Summary": r"(Summary|Товч танилцуулга|overview)",
        "Languages": r"(Languages|Хэл|linguistic)",
        "Contact": r"(Contact|Мэдээлэл|info|profile)"
    }
    return {k: bool(re.search(v, text, re.IGNORECASE)) for k, v in section_keywords.items()}


def extract_resume_sections(text: str) -> Dict[str, str]:
    """Split resume into sections with their content."""
    section_patterns = {
        'contact': r'(contact|personal|info|information|profile|details|мэдээлэл)',
        'education': r'(education|academic|qualification|degree|university|school|боловсрол|сургалт)',
        'experience': r'(experience|employment|work|history|professional|туршлага|ажилласан)',
        'skills': r'(skills|abilities|expertise|competencies|proficiencies|чадвар)',
        'projects': r'(projects|portfolio|works|төслүүд)',
        'achievements': r'(achievements|accomplishments|honors|awards)',
        'languages': r'(languages|linguistic|хэл)',
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


def extract_top_skills(text: str, top_n: int = 15) -> List[str]:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = {
        'the', 'and', 'for', 'with', 'that', 'have', 'this', 'from', 'you', 'but', 'are',
        'all', 'can', 'was', 'has', 'will', 'not', 'who', 'your', 'their', 'experience', 'education'
    }
    common = [w for w in words if w not in stop_words]
    return [word for word, _ in Counter(common).most_common(top_n)]


def extract_roles(text: str) -> List[str]:
    keywords = [
        "teacher", "manager", "coordinator", "developer", "designer", "engineer",
        "service", "consultant", "director", "analyst", "specialist",
        "менежер", "багш", "захирал", "туслах", "сургалт", "хариуцсан"
    ]
    found_roles = [kw for kw in keywords if re.search(rf"\b{kw}\b", text, re.IGNORECASE)]
    return list(set(found_roles))


def extract_contact_info(text: str) -> Dict[str, Any]:
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    phone = re.findall(r'(\+?[\d\s()-]{7,})', text)
    linkedin = re.findall(r'linkedin\.com/in/[\w-]+', text)
    locations = [ent.text for ent in nlp(text).ents if ent.label_ == "GPE"]
    return {
        'email': email,
        'phone': phone,
        'linkedin': linkedin,
        'location': locations
    }


def analyze_resume(text: str) -> Dict[str, Any]:
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
        'skills': extract_top_skills(text),
        'roles': extract_roles(text),
        'completeness_score': completeness_score
    }


def summarize_resume(text: str) -> Dict[str, Any]:
    years_exp = len(set(re.findall(r"\b(20[0-9]{2})\b", text)))
    roles = extract_roles(text)
    skills = extract_top_skills(text, 10)
    return {
        "years_of_experience": years_exp,
        "roles": roles,
        "top_skills": skills
    }
