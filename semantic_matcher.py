import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import spacy
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load spaCy model for NER
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# Initialize sentence transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def extract_resume_keywords(text, top_n=30):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    stop_words = set(stopwords.words('english'))
    mongolian_stopwords = {"ба", "болон", "мөн", "юм", "бол", "нь", "л", "тэр", "энэ", "би", "та", "тэд"}
    stop_words.update(mongolian_stopwords)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE"]]
    word_freq = Counter(tokens + entities)
    return [word for word, _ in word_freq.most_common(top_n)]

def extract_skills_from_resume(text):
    tech_skills = ["python", "java", "c++", "javascript", "html", "css", "sql", "nosql",
                   "mongodb", "react", "angular", "vue", "node", "express", "django",
                   "flask", "tensorflow", "pytorch", "ai", "ml", "data science",
                   "docker", "kubernetes", "aws", "azure", "gcp", "git", "devops"]
    soft_skills = ["communication", "teamwork", "leadership", "problem solving",
                   "critical thinking", "time management", "creativity", "adaptability",
                   "project management", "analytical", "detail oriented"]
    mongolian_skills = ["монгол хэл", "орос хэл", "англи хэл", "хятад хэл", "солонгос хэл",
                        "япон хэл", "удирдлага", "менежмент", "санхүү", "нягтлан бодох"]
    all_skills = tech_skills + soft_skills + mongolian_skills
    text_lower = text.lower()
    found_skills = [skill for skill in all_skills if skill in text_lower]
    doc = nlp(text)
    found_skills += [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    return list(set(found_skills))

def semantic_match_resume(resume_text, jobs_df, top_n=10):
    jobs_df["combined_text"] = (
        jobs_df["Job title"].fillna('') + " " +
        jobs_df["Job description"].fillna('') + " " +
        jobs_df["Requirements"].fillna('')
    )
    resume_embedding = model.encode([resume_text])[0]
    job_embeddings = model.encode(jobs_df["combined_text"].tolist())
    similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0]
    result_df = jobs_df.copy()
    result_df["match_score"] = similarity_scores * 100
    return result_df.sort_values(by="match_score", ascending=False).head(top_n)

def get_skill_matches(resume_skills, job_text):
    job_text_lower = job_text.lower()
    matched_skills = [skill for skill in resume_skills if skill in job_text_lower]
    job_skills = extract_skills_from_resume(job_text)
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    return matched_skills, missing_skills

def analyze_resume(resume_text):
    skills = extract_skills_from_resume(resume_text)
    keywords = extract_resume_keywords(resume_text)
    sections = {
        "education": len(re.findall(r'education|degree|university|college|school', resume_text.lower())),
        "experience": len(re.findall(r'experience|work|job|position|career', resume_text.lower())),
        "skills": len(re.findall(r'skills|abilities|proficient|expertise', resume_text.lower())),
        "projects": len(re.findall(r'project|portfolio|developed|created|built', resume_text.lower())),
    }
    completeness = min(100, (sum(sections.values()) / 8) * 100)
    return {
        "skills": skills,
        "keywords": keywords,
        "sections": sections,
        "completeness": completeness
    }
