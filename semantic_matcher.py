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
import streamlit as st # Import streamlit for caching

logger = logging.getLogger(__name__)

# --- NLTK Data Downloads ---
# Ensure NLTK data is available. Downloads if not found.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt (semantic_matcher)...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords (semantic_matcher)...")
    nltk.download('stopwords', quiet=True)


# --- SpaCy Model Loading ---
# Load spaCy model (v3+ compatible)
# Use @st.cache_resource to load the model only once
@st.cache_resource
def load_spacy_model_matcher(model_name):
    """Loads a spaCy model for the matcher and caches it."""
    try:
        logger.info(f"Attempting to load spaCy model '{model_name}' for semantic matching.")
        nlp_model = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model '{model_name}' for semantic matching.")
        return nlp_model
    except OSError as e:
        error_msg = f"FATAL ERROR: Could not load spaCy model '{model_name}' in semantic_matcher. Ensure it is installed."
        logger.error(error_msg, exc_info=True)
        raise

nlp_matcher = load_spacy_model_matcher("en_core_web_sm")


# --- Sentence Transformer Model Loading ---
# Load the sentence transformer model - installed via requirements.txt
# Use @st.cache_resource to load this large model only once
@st.cache_resource
def load_sentence_transformer(model_name):
    """Loads a SentenceTransformer model and caches it."""
    try:
        logger.info(f"Loading SentenceTransformer model '{model_name}'")
        model = SentenceTransformer(model_name)
        logger.info(f"SentenceTransformer model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
         logger.error(f"Error loading SentenceTransformer model '{model_name}': {e}", exc_info=True)
         raise # Re-raise if model loading is critical


model = load_sentence_transformer('paraphrase-multilingual-MiniLM-L12-v2')


# --- Keyword/Skill Extraction for Matching ---
# These are specific extraction methods used by the semantic matching logic

def extract_resume_keywords(text, top_n=30):
    """Extracts keywords from text using tokenization, stopwords, and spaCy entities."""
    text = re.sub(r'[^\w\s]', ' ', text.lower()) # Clean and lowercase
    stop_words = set(stopwords.words('english'))
    # Added Mongolian stopwords from your old code
    mongolian_stopwords = {"ба", "болон", "мөн", "юм", "бол", "нь", "л", "тэр", "энэ", "би", "та", "тэд"}
    stop_words.update(mongolian_stopwords)

    tokenizer = RegexpTokenizer(r'\w+') # Tokenizer for words
    tokens = tokenizer.tokenize(text)
    # Filter out non-alpha words, short words, and stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]

    doc = nlp_matcher(text) # Use the cached spaCy model
    # Extract specific entity types (ORG, PRODUCT, LANGUAGE, SKILL as determined in earlier debugging)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "SKILL"]]

    # Combine tokens and entities, count frequency
    word_freq = Counter(tokens + entities)
    # Return top_n most common words/entities
    return [word for word, _ in word_freq.most_common(top_n)]

def extract_skills_from_text(text):
    """Extracts skills from text based on predefined lists and spaCy entities (Semantic method)."""
    # Defined skill lists (expanded with Mongolian skills from your old code)
    tech_skills = ["python", "java", "c++", "javascript", "html", "css", "sql", "nosql",
                   "mongodb", "react", "angular", "vue", "node", "express", "django",
                   "flask", "tensorflow", "pytorch", "ai", "ml", "data science",
                   "docker", "kubernetes", "aws", "azure", "gcp", "git", "devops"]
    soft_skills = ["communication", "teamwork", "leadership", "problem solving",
                   "critical thinking", "time management", "creativity", "adaptability",
                   "project management", "analytical", "detail oriented"]
    # Added Mongolian skills from your old code
    mongolian_skills = ["монгол хэл", "орос хэл", "англи хэл", "хятад хэл", "солонгос хэл",
                        "япон хэл", "удирдлага", "менежмент", "санхүү", "нягтлан бодох"]
    all_skills = tech_skills + soft_skills + mongolian_skills

    text_lower = text.lower()
    found_skills = [skill for skill in all_skills if skill in text_lower]

    doc = nlp_matcher(text) # Use the cached spaCy model
    # Extract specific entity types (ORG, PRODUCT, SKILL as determined in earlier debugging)
    found_skills += [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]

    return list(set(found_skills))


# --- Semantic Matching Function ---

def semantic_match_resume(resume_text, jobs_df, top_n=10):
    """Performs semantic matching between resume text and job descriptions in a DataFrame."""
    logger.info("Starting semantic matching.")
    # Ensure required columns exist and handle NaNs - Match columns expected by your old app.py
    required_cols = ["Job title", "Job description", "Requirements"]
    for col in required_cols:
        if col not in jobs_df.columns:
            logger.error(f"Missing required column for semantic matching: {col}. Cannot perform match.")
            # Return empty df on error, but keep the original columns + match_score for display readiness
            return pd.DataFrame(columns=jobs_df.columns.tolist() + ['match_score'])

    # Combine relevant job columns into a single text field for embedding
    jobs_df["combined_text"] = (
        jobs_df["Job title"].fillna('').astype(str) + " " +
        jobs_df["Job description"].fillna('').astype(str) + " " +
        jobs_df["Requirements"].fillna('').astype(str)
    )

    try:
        # Encode resume and job texts into vectors using the cached model
        # Use convert_to_tensor=True and handle CPU transfer for cosine_similarity as in previous working version
        resume_embedding = model.encode([resume_text], convert_to_tensor=True)[0]
        job_embeddings = model.encode(jobs_df["combined_text"].tolist(), convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_scores = cosine_similarity([resume_embedding.cpu()], job_embeddings.cpu())[0] * 100 # Move to CPU for sklearn

        # Add scores to a copy of the DataFrame and sort
        result_df = jobs_df.copy()
        result_df["match_score"] = similarity_scores

        logger.info("Semantic matching complete.")
        return result_df.sort_values(by="match_score", ascending=False).head(top_n)

    except Exception as e:
        logger.error(f"Error during semantic matching: {e}", exc_info=True)
        # Return empty df on error, but keep the original columns + match_score
        return pd.DataFrame(columns=jobs_df.columns.tolist() + ['match_score'])


# --- Skill Matching Function (Used by the simple matching tab) ---
# This function compares resume skills (from the basic parser) against job skills
# extracted by the semantic method's extract_skills_from_text

def get_skill_matches(resume_skills_basic, job_text):
    """Compares a list of resume skills (basic) against skills found in job text (semantic method)."""
    logger.info("Getting skill matches (semantic_matcher method).")

    # Extract skills from the job description using the method defined in *this* file (semantic_matcher)
    job_skills_semantic = extract_skills_from_text(job_text)
    job_skills_semantic_lower = [s.lower() for s in job_skills_semantic]

    # Convert the input resume skills list (from basic parser) to lowercase for comparison
    resume_skills_basic_lower = [s.lower() for s in resume_skills_basic]

    # Find skills present in both lists (case-insensitive comparison)
    matched_skills_lower = set(resume_skills_basic_lower).intersection(set(job_skills_semantic_lower))

    # Find skills in the job description that are NOT in the resume skills list
    missing_skills_lower = set(job_skills_semantic_lower).difference(set(resume_skills_basic_lower))

    # Convert matched and missing skills back to their original casing from the *job skills* list
    matched_skills_original_case = [s for s in job_skills_semantic if s.lower() in matched_skills_lower]
    missing_skills_original_case = [s for s in job_skills_semantic if s.lower() in missing_skills_lower]

    logger.info(f"Found {len(matched_skills_original_case)} matching skills and {len(missing_skills_original_case)} missing skills.")
    return matched_skills_original_case, missing_skills_original_case


# --- analyze_resume (Semantic version logic, as seen in your old code) ---
# This function is NOT imported by the app.py provided, which imported 'analyze_resume'
# from resume_parser and aliased 'analyze_resume' from semantic_matcher as 'analyze_resume_skills'.
# However, the 'Resume Analysis' tab in your old app.py *does* call analyze_resume from resume_parser
# and analyze_resume_skills from semantic_matcher (which is THIS function aliased).
# So, this function name needs to be accessible as 'analyze_resume' within this module
# so the old app.py's import as 'analyze_resume_skills' works.

def analyze_resume(resume_text):
    """Analyzes resume text for keywords, skills (using method in this file), and section presence (Semantic version)."""
    # This function name conflicts with resume_parser.analyze_resume, but the old app.py
    # explicitly imported both (one directly, one aliased)
    logger.info("Starting semantic resume analysis (keywords, skills, sections by keyword).")

    # Use the skill extraction method defined in *this* file
    skills = extract_skills_from_text(resume_text)
    keywords = extract_resume_keywords(resume_text)

    # Basic keyword check for sections (as in your old code)
    sections = {
        "education": len(re.findall(r'education|degree|university|college|school', resume_text.lower())),
        "experience": len(re.findall(r'experience|work|job|position|career', resume_text.lower())), # Matches your old regex
        "skills": len(re.findall(r'skills|abilities|proficient|expertise', resume_text.lower())), # Matches your old regex
        "projects": len(re.findall(r'project|portfolio|developed|created|built', resume_text.lower())),
        # Your old code only checked for these 4 sections in the regex findall part
        # "contact": len(re.findall(r'contact|email|phone|linkedin', resume_text.lower())),
        # "summary": len(re.findall(r'summary|objective|profile', resume_text.lower())),
    }

    # Calculate completeness based on sum of counts divided by a hardcoded 8 (as in your old code)
    # Note: Your old code divided the *sum of counts* by 8, which results in a different score
    # than checking *how many* sections had >0 keywords and dividing by the number of sections checked.
    # Let's replicate your old code's calculation exactly for completeness.
    completeness = min(100, (sum(sections.values()) / 8) * 100)


    logger.info("Semantic resume analysis complete.")
    return {
        "skills": skills,
        "keywords": keywords,
        "sections": sections, # Renamed from sections_keyword_count to match your old code
        "completeness": completeness # Renamed from completeness_score to match your old code
    }

# Expose models for debug check in app.py if needed
nlp_matcher_model_for_debug = nlp_matcher
st_model_for_debug = model
