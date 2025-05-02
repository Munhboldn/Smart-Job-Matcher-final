import fitz  # PyMuPDF
import docx2txt
import tempfile
import re
import os
from docx import Document
import spacy

# Load spaCy model for NER (optional)
try:
    nlp = spacy.load('en_core_web_sm')
except:
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load('en_core_web_sm')
    except Exception as e:
        print(f"spaCy load error: {e}")
        nlp = None


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_file.seek(0)
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda block: (block[1], block[0]))
                for block in blocks:
                    block_text = block[4].strip()
                    if block_text:
                        text += block_text + "\n\n"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
    return clean_resume_text(text)


def extract_text_from_docx(docx_file):
    try:
        docx_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(docx_file.read())
            tmp_path = tmp.name

        doc = Document(tmp_path)
        full_text = []

        for section in doc.sections:
            for header in section.header.paragraphs:
                if header.text.strip():
                    full_text.append(header.text.strip())

        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))

        os.unlink(tmp_path)
        return clean_resume_text("\n\n".join(full_text))

    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"


def clean_resume_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.replace('•', '\n• ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_resume_sections(text):
    section_patterns = {
        'contact': r'^\s*(contact|personal|info|information|profile|details)\s*$',
        'summary': r'^\s*(summary|objective|about)\s*$',
        'education': r'^\s*(education|academic|qualification|degree|university|school)\s*$',
        'experience': r'^\s*(experience|employment|work|history|professional|career)\s*$',
        'skills': r'^\s*(skills|abilities|expertise|competencies|technical skills)\s*$'
    }

    sections = {}
    current_section = None
    current_text = []

    lines = text.split('\n')

    for line in lines:
        header_found = False
        for section, pattern in section_patterns.items():
            if re.match(pattern, line.strip(), re.IGNORECASE):
                if current_section:
                    sections[current_section] = '\n'.join(current_text).strip()
                current_section = section
                current_text = []
                header_found = True
                break
        if not header_found and current_section:
            current_text.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_text).strip()

    return sections


def extract_contact_info(text):
    return {
        "emails": re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text),
        "phones": re.findall(r'\+?\d[\d\s().-]{7,}', text),
        "linkedin": re.findall(r'(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9\-_/]+', text),
        "locations": re.findall(r'[A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)+', text),
        "name": re.findall(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text.strip().split('\n')[0])
    }


def analyze_resume(text):
    sections = extract_resume_sections(text)
    expected = ['contact', 'summary', 'education', 'experience', 'skills']
    found = [s for s in expected if s in sections and len(sections[s].strip()) > 30]
    completeness = int((len(found) / len(expected)) * 100)

    return {
        "sections": sections,
        "completeness_score": completeness,
        "contact_info": extract_contact_info(text)
    }
