import pandas as pd
import numpy as np
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import spacy
from fuzzywuzzy import fuzz
from termcolor import colored
from pathlib import Path

# --- Setup ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# --- Clean Resume Text ---
def clean_resume_text(text):
    text = re.sub(r'(name|contact|email|phone|address)[\s:]+.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Extract Keywords ---
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

# --- Fuzzy Match Skills ---
def fuzzy_match_skills(text, skill_list, threshold=80):
    found = []
    for skill in skill_list:
        if fuzz.partial_ratio(skill, text.lower()) >= threshold:
            found.append(skill)
    return list(set(found))

# --- Extract Skills ---
def extract_skills_from_resume(text):
    skill_list = [
        "python", "java", "c++", "javascript", "html", "css", "sql", "nosql", "mongodb", "react", "angular",
        "vue", "node", "express", "django", "flask", "tensorflow", "pytorch", "ai", "ml", "data science",
        "docker", "kubernetes", "aws", "azure", "gcp", "git", "devops",
        "communication", "teamwork", "leadership", "problem solving", "critical thinking",
        "time management", "creativity", "adaptability", "project management", "analytical",
        "detail oriented",
        "монгол хэл", "орос хэл", "англи хэл", "хятад хэл", "солонгос хэл",
        "япон хэл", "удирдлага", "менежмент", "санхүү", "нягтлан бодох"
    ]
    return fuzzy_match_skills(text, skill_list)

# --- Mongolian NER (rule-based) ---
def extract_mongolian_entities(text):
    known_entities = ["Улаанбаатар", "МУИС", "ХААИС", "Боловсрол", "Санхүү", "Хууль"]
    return [ent for ent in known_entities if ent.lower() in text.lower()]

# --- Highlight Skills ---
def highlight_matched_skills(text, skills):
    highlighted = text
    for skill in sorted(set(skills), key=len, reverse=True):
        pattern = re.compile(re.escape(skill), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: colored(m.group(), 'green', attrs=['bold']), highlighted)
    return highlighted

# --- Save CSV ---
def save_results_to_csv(results_df, filename="matched_jobs.csv"):
    Path("output").mkdir(exist_ok=True)
    results_df.to_csv(f"output/{filename}", index=False)
    print(f"✅ Results saved to output/{filename}")

# --- Semantic Matcher ---
def semantic_match_resume(resume_text, jobs_df, top_n=10, min_score=40, save_csv=True, highlight=False):
    cleaned_text = clean_resume_text(resume_text)
    keywords = extract_resume_keywords(cleaned_text)
    skills = extract_skills_from_resume(cleaned_text)
    mongolian_ents = extract_mongolian_entities(cleaned_text)
    enriched_resume = cleaned_text + " " + " ".join(keywords + skills + mongolian_ents)

    jobs_df["combined_text"] = (
        jobs_df["Job title"].fillna('') * 2 + " " +
        jobs_df["Job description"].fillna('') + " " +
        jobs_df["Requirements"].fillna('')
    )

    resume_embedding = model.encode([enriched_resume])[0]
    job_embeddings = model.encode(jobs_df["combined_text"].tolist())
    similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0]

    result_df = jobs_df.copy()
    result_df["match_score"] = similarity_scores * 100
    result_df = result_df[result_df["match_score"] >= min_score]
    result_df = result_df.sort_values(by="match_score", ascending=False).head(top_n)

    print(result_df[["Job title", "match_score"]])

    if highlight:
        for _, row in result_df.iterrows():
            print("\n🧾", row["Job title"], f"[{row['match_score']:.2f}% match]")
            highlighted_desc = highlight_matched_skills(row["Job description"], skills)
            print(highlighted_desc)

    if save_csv:
        save_results_to_csv(result_df)

    return result_df

# --- Resume Analyzer ---
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

# --- Skill Match Checker ---
def get_skill_matches(resume_skills, job_text):
    job_text_lower = job_text.lower()
    matched_skills = [skill for skill in resume_skills if skill in job_text_lower]
    job_skills = extract_skills_from_resume(job_text)
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    return matched_skills, missing_skills
