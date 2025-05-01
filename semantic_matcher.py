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

# --- NLTK and spaCy Resources ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_transformer():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

nlp = load_spacy_model()
model = load_transformer()

# --- Keyword Extraction ---
def extract_resume_keywords(text, top_n=30):
    text = re.sub(r'[^Ѐ-ӿ\w\s]', ' ', text.lower())
    stop_words = set(stopwords.words('english'))
    stop_words.update({
        "ба", "болон", "мөн", "юм", "бол", "нь", "л", "тэр", "энэ", "би", "та", "тэд",
        "байна", "хийдэг", "гадаад", "ажил", "туршлага", "компани", "сургууль",
        "мэргэжил", "он", "сарын", "өдөр", "байгууллага", "хөтөлбөр"
    })
    tokens = [word for word in RegexpTokenizer(r'\w+").tokenize(text) if word.isalpha() and word not in stop_words and len(word) > 2]
    ents = [ent.text.lower() for ent in nlp(text).ents if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "SKILL"]]
    return [word for word, _ in Counter(tokens + ents).most_common(top_n)]

# --- Skill Extraction ---
def extract_skills_from_text(text, job_corpus=None):
    """
    Extracts skills by matching against a curated list and named entities.
    You can expand this by combining it with job dataset-derived phrases.
    """
    tech_skills = [
        "python", "java", "c++", "c#", "sql", "nosql", "html", "css", "javascript",
        "react", "angular", "vue", "node", "django", "flask", "spring", "docker", "kubernetes",
        "aws", "azure", "gcp", "linux", "git", "devops", "mongodb", "postgresql"
    ]

    soft_skills = [
        "teamwork", "leadership", "communication", "problem solving", "adaptability",
        "project management", "time management", "critical thinking", "creativity",
        "collaboration", "analytical skills", "attention to detail"
    ]

    mongolian_skills = [
        "монгол хэл", "англи хэл", "орос хэл", "хятад хэл", "солонгос хэл", "япон хэл",
        "нягтлан", "удирдлага", "сургалт", "багш", "борлуулалт", "үйлчилгээ",
        "менежмент", "санхүү", "маркетинг", "төсөл", "судалгаа"
    ]

    all_skills = tech_skills + soft_skills + mongolian_skills

    # Expand with dynamic job corpus phrases
    if job_corpus is not None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(job_corpus.dropna().astype(str))
        dynamic_keywords = tfidf.get_feature_names_out()
        all_skills += list(dynamic_keywords)
    text_lower = text.lower()
    found = [skill for skill in all_skills if skill in text_lower]
    ents = [ent.text.lower() for ent in nlp(text).ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]
    return list(set(found + ents))

# --- Semantic Matching ---
def semantic_match_resume(resume_text, jobs_df, top_n=10):
    if not all(col in jobs_df.columns for col in ["Job title", "Job description", "Requirements"]):
        logger.warning("Required job columns missing")
        return pd.DataFrame(columns=list(jobs_df.columns) + ['match_score'])

    jobs_df["combined_text"] = jobs_df["Job title"].fillna('') + ", " + jobs_df["Job description"].fillna('') + ", " + jobs_df["Requirements"].fillna('')
    job_embeddings = model.encode(jobs_df["combined_text"].tolist())
    resume_embedding = model.encode([resume_text])[0]
    scores = cosine_similarity([resume_embedding], job_embeddings)[0] * 100
    jobs_df["match_score"] = scores
        jobs_df = jobs_df.sort_values("match_score", ascending=False)
    jobs_df["matched_skills"], jobs_df["missing_skills"] = zip(*jobs_df["combined_text"].apply(lambda txt: get_skill_matches(extract_skills_from_text(resume_text, jobs_df["combined_text"]), txt)))
    return jobs_df.head(top_n)

# --- Skill Matching ---
def get_skill_matches(resume_skills, job_text):
    job_skills = extract_skills_from_text(job_text)
    resume_skills_lower = set(skill.lower() for skill in resume_skills)
    matched = [skill for skill in job_skills if skill.lower() in resume_skills_lower]
    missing = [skill for skill in job_skills if skill.lower() not in resume_skills_lower]
    return matched, missing

# --- Resume Analysis ---
def analyze_resume(resume_text):
    """
    Analyze the resume by extracting:
    - Recognized skills
    - Top keywords
    - Section presence (via regex)
    - Completeness score (based on presence count)
    """
    skills = extract_skills_from_text(resume_text)
    keywords = extract_resume_keywords(resume_text)
    section_patterns = {
        "education": r"education|degree|university|college|school",
        "experience": r"experience|work|job|position|career",
        "skills": r"skills|abilities|proficient|expertise",
        "projects": r"project|portfolio|developed|created|built"
    }
    sections = {sec: len(re.findall(pattern, resume_text.lower())) for sec, pattern in section_patterns.items()}
    completeness = min(100, (sum(sections.values()) / 8) * 100)
    return {
        "skills": skills,
        "keywords": keywords,
        "sections": sections,
        "completeness": completeness
    }
