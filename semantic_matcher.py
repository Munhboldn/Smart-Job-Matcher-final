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
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# --- NLTK Data Downloads ---
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

# --- SpaCy Model Loading with Streamlit Caching ---
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        logger.info(f"Loading spaCy model '{model_name}'")
        model = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model '{model_name}'")
        return model
    except OSError:
        logger.error(f"Could not load spaCy model '{model_name}'. Ensure it's installed.", exc_info=True)
        raise

nlp = load_spacy_model()

# --- Sentence Transformer Model Loading with Streamlit Caching ---
@st.cache_resource
def load_sentence_transformer(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    try:
        logger.info(f"Loading SentenceTransformer model '{model_name}'")
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded SentenceTransformer model '{model_name}'")
        return model
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model '{model_name}': {e}", exc_info=True)
        raise

model = load_sentence_transformer()

# --- Keyword Extraction ---
def extract_resume_keywords(text, top_n=30):
    text = re.sub(r'[^Ѐ-\u04FF\w\s]', ' ', text.lower())
    stop_words = set(stopwords.words('english'))
    mongolian_stopwords = {"ба", "болон", "мөн", "юм", "бол", "нь", "л", "тэр", "энэ", "би", "та", "тэд"}
    stop_words.update(mongolian_stopwords)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [word for word in tokenizer.tokenize(text) if word.isalpha() and word not in stop_words and len(word) > 2]

    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "SKILL"]]

    word_freq = Counter(tokens + entities)
    return [word for word, _ in word_freq.most_common(top_n)]

# --- Skill Extraction ---
def extract_skills_from_text(text):
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
    found_skills += [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]

    return list(set(found_skills))

# --- Semantic Matching ---
def semantic_match_resume(resume_text, jobs_df, top_n=10):
    required_cols = ["Job title", "Job description", "Requirements"]
    for col in required_cols:
        if col not in jobs_df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame(columns=jobs_df.columns.tolist() + ['match_score'])

    jobs_df["combined_text"] = (
        jobs_df["Job title"].fillna('') + " " +
        jobs_df["Job description"].fillna('') + " " +
        jobs_df["Requirements"].fillna('')
    )

    resume_embedding = model.encode([resume_text])[0]
    job_embeddings = model.encode(jobs_df["combined_text"].tolist())

    similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0] * 100
    result_df = jobs_df.copy()
    result_df["match_score"] = similarity_scores

    return result_df.sort_values(by="match_score", ascending=False).head(top_n)

# --- Skill Matching ---
def get_skill_matches(resume_skills, job_text):
    job_skills = extract_skills_from_text(job_text)
    resume_skills_lower = [skill.lower() for skill in resume_skills]

    matched_skills = [skill for skill in job_skills if skill.lower() in resume_skills_lower]
    missing_skills = [skill for skill in job_skills if skill.lower() not in resume_skills_lower]

    return matched_skills, missing_skills

# --- Resume Analysis ---
def analyze_resume(resume_text):
    skills = extract_skills_from_text(resume_text)
    keywords = extract_resume_keywords(resume_text)

    sections = {
        "education": len(re.findall(r'education|degree|university|college|school', resume_text.lower())),
        "experience": len(re.findall(r'experience|work|job|position|career', resume_text.lower())),
        "skills": len(re.findall(r'skills|abilities|proficient|expertise', resume_text.lower())),
        "projects": len(re.findall(r'project|portfolio|developed|created|built', resume_text.lower()))
    }

    completeness = min(100, (sum(sections.values()) / 8) * 100)

    return {
        "skills": skills,
        "keywords": keywords,
        "sections": sections,
        "completeness": completeness
    }
