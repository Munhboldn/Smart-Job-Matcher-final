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
import os # Import os for model path

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK data not found. Downloading punkt and stopwords...")
    nltk.download('punkt')
    nltk.download('stopwords')
    print("NLTK data download complete.")


# Load spaCy model for NER
# Check if model is already downloaded or try downloading
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("spaCy model 'en_core_web_sm' not found. Attempting download...")
    try:
        import subprocess
        # Use --user to install in user site-packages if necessary
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load('en_core_web_sm')
        print("spaCy model downloaded and loaded successfully.")
    except Exception as e:
        print(f"Error loading or downloading spaCy model: {e}")
        print("Proceeding without spaCy NER features for skill/keyword extraction.")
        nlp = None # Set nlp to None if loading fails


# Initialize sentence transformer model
# Check for local model first before downloading
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = None
try:
    # Attempt to load from default cache location
    model = SentenceTransformer(model_name)
    print(f"SentenceTransformer model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model '{model_name}': {e}")
    print(f"Attempting to download model '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
        print(f"SentenceTransformer model '{model_name}' downloaded and loaded successfully.")
    except Exception as e_download:
        print(f"Failed to download SentenceTransformer model '{model_name}': {e_download}")
        print("Semantic matching will not be available.")
        model = None # Set model to None if download fails


def preprocess_text(text):
    """Basic text cleaning for keyword/skill extraction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove special characters but keep spaces and hyphens for phrases
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_resume_keywords(text, top_n=50): # Increased top_n
    """Extracts keywords from resume text."""
    processed_text = preprocess_text(text)
    if not processed_text:
        return []

    stop_words = set(stopwords.words('english'))
    # Added more common Mongolian stop words
    mongolian_stopwords = {"ба", "болон", "мөн", "юм", "бол", "нь", "л", "тэр", "энэ", "би", "та", "тэд",
                           "гэх", "нь", "үед", "байгаа", "гэж", "дээр", "дотор", "тухай", "харин", "гэсэн"}
    stop_words.update(mongolian_stopwords)

    tokenizer = RegexpTokenizer(r'\w+') # Tokenize into words
    tokens = tokenizer.tokenize(processed_text)

    # Filter tokens
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2] # Filter short words and stop words

    keywords = tokens # Start with individual words

    # Add multi-word phrases from N-grams (optional, can add noise)
    # from nltk.util import ngrams
    # bigrams = [' '.join(grams) for grams in ngrams(tokens, 2) if ' '.join(grams) not in stop_words]
    # trigrams = [' '.join(grams) for grams in ngrams(tokens, 3) if ' '.join(grams) not in stop_words]
    # keywords.extend(bigrams + trigrams)


    # Add entities from spaCy if available
    if nlp:
        doc = nlp(processed_text)
        # Consider more relevant entity types like ORG, PRODUCT, GPE, NORP, FAC, LOC
        # Filter out common short entities that might not be keywords
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE", "NORP", "FAC", "LOC", "PERSON"] and len(ent.text.split()) > 1 and len(ent.text) > 3] # Filter short entities and single words

        # Also consider noun chunks as potential keywords
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1 and chunk.text.lower() not in stop_words]
        keywords.extend(entities + noun_chunks)


    # Count frequency and get top N
    word_freq = Counter(keywords)
    # Filter out keywords that are just numbers
    top_keywords = [word for word, _ in word_freq.most_common(top_n) if not word.isdigit()]

    return top_keywords


def extract_skills_from_resume(text):
    """Extracts predefined and potentially entity-based skills from text."""
    processed_text = preprocess_text(text)
    if not processed_text:
        return []

    # Expanded and refined skill lists
    tech_skills = [
        "python", "java", "c++", "javascript", "html", "css", "sql", "nosql",
        "mongodb", "react", "angular", "vue", "node", "express", "django",
        "flask", "tensorflow", "pytorch", "keras", "scikit-learn", "numpy", "pandas",
        "ai", "ml", "machine learning", "data science", "data analysis", "big data",
        "docker", "kubernetes", "aws", "azure", "gcp", "git", "devops", "cloud computing",
        "linux", "unix", "windows server", "networking", "cybersecurity", "database management",
        "web development", "mobile development", "api development", "software engineering",
        "quality assurance", "ui/ux design", "graphic design", "photoshop", "illustrator", "figma",
        "excel", "word", "powerpoint", "microsoft office", "google workspace"
    ]
    soft_skills = [
        "communication", "teamwork", "collaboration", "leadership", "problem solving",
        "critical thinking", "time management", "organization", "creativity", "adaptability",
        "flexibility", "project management", "analytical skills", "detail oriented",
        "customer service", "client management", "negotiation", "presentation skills",
        "interpersonal skills", "mentoring", "training", "coaching"
    ]
    mongolian_skills = [
        "монгол хэл", "орос хэл", "англи хэл", "хятад хэл", "солонгос хэл", "япон хэл",
        "удирдлага", "менежмент", "санхүү", "нягтлан бодох", "борлуулалт", "маркетинг",
        "хүний нөөц", "хууль", "логистик", "хангамж", "үйлчилгээ", "багшлах", "сургалт"
    ]

    all_skills = set(tech_skills + soft_skills + mongolian_skills) # Use a set for faster lookup

    found_skills = []
    # Check for presence of predefined skills
    for skill in all_skills:
        if skill in processed_text:
            found_skills.append(skill)

    # Add entities from spaCy if available, focusing on relevant types that might be skills/technologies
    if nlp:
        doc = nlp(processed_text)
        entity_skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "TECH"] and len(ent.text) > 2] # Added "TECH" if your model supports it, filter short ones
        found_skills.extend(entity_skills)

    # Use a set to get unique skills and convert back to list
    return list(set(found_skills))


def semantic_match_resume(resume_text, jobs_df, top_n=10):
    """
    Performs semantic similarity matching between a resume and job descriptions.
    Requires the SentenceTransformer model to be loaded.
    """
    if model is None:
        print("SentenceTransformer model not loaded. Cannot perform semantic matching.")
        return pd.DataFrame() # Return empty DataFrame if model is not available

    # Ensure necessary columns exist
    required_cols = ["Job title", "Job description", "Requirements", "Company", "Salary", "URL"]
    for col in required_cols:
        if col not in jobs_df.columns:
            print(f"Warning: Missing required column '{col}' in jobs DataFrame.")
            jobs_df[col] = "" # Add missing column with empty strings

    # Combine relevant job text for embedding
    # Handle potential non-string types before combining
    jobs_df["combined_text"] = (
        jobs_df["Job title"].astype(str).fillna('') + " " +
        jobs_df["Job description"].astype(str).fillna('') + " " +
        jobs_df["Requirements"].astype(str).fillna('')
    ).apply(preprocess_text) # Apply preprocessing

    # Ensure resume_text is a string and preprocess
    resume_text_processed = preprocess_text(resume_text)
    if not resume_text_processed:
         print("Warning: Resume text is empty after preprocessing. Cannot perform matching.")
         return pd.DataFrame()

    try:
        # Encode the resume and job texts
        resume_embedding = model.encode([resume_text_processed])[0]
        # Filter out empty job texts before encoding
        valid_job_texts = jobs_df[jobs_df["combined_text"].str.len() > 0]["combined_text"].tolist()
        valid_job_indices = jobs_df[jobs_df["combined_text"].str.len() > 0].index

        if not valid_job_texts:
             print("No valid job texts found for encoding.")
             return pd.DataFrame()

        job_embeddings = model.encode(valid_job_texts)

        # Calculate cosine similarity
        similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0]

        # Create a result DataFrame and map scores back to original jobs_df indices
        result_df = jobs_df.loc[valid_job_indices].copy()
        result_df["match_score"] = similarity_scores * 100

        # Sort and return top N results
        return result_df.sort_values(by="match_score", ascending=False).head(top_n)

    except Exception as e:
        print(f"Error during semantic matching: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


def get_skill_matches(resume_skills, job_text):
    """
    Compares skills extracted from a resume against skills/keywords in a job description.
    """
    if not resume_skills:
        return [], [] # Return empty lists if no resume skills are provided

    job_text_processed = preprocess_text(job_text)
    if not job_text_processed:
         return [], []

    # Extract skills and keywords from the job text
    job_skills = extract_skills_from_resume(job_text_processed)
    job_keywords = extract_resume_keywords(job_text_processed, top_n=100) # Get more keywords from job

    # Combine job skills and keywords for comparison
    job_terms = set(job_skills + job_keywords)

    # Find matched skills (skills from resume present in job terms)
    # Ensure comparison is case-insensitive
    resume_skills_lower = [s.lower() for s in resume_skills]
    matched_skills = [skill for skill in resume_skills if skill.lower() in job_terms]

    # Find potentially missing skills (skills/keywords from job terms not in resume skills)
    # This is a simplification; a missing "keyword" might not be a "skill" you need to add.
    # It's better to focus on missing skills from the predefined list or job-specific entities.
    # Let's redefine missing skills as skills/entities found in the job text but not in resume skills.
    job_entities = []
    if nlp:
         doc = nlp(job_text_processed)
         job_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "TECH"] and len(ent.text) > 2]

    all_job_relevant_terms = set(job_skills + job_entities)

    # Compare lowercase job terms against lowercase resume skills
    missing_skills = [term for term in all_job_relevant_terms if term not in resume_skills_lower]


    # Optional: Refine missing skills to be more relevant (e.g., filter out very common words)
    # This requires a more sophisticated approach, potentially comparing embeddings or using a skill taxonomy.
    # For now, the current approach identifies terms in the job that weren't in your resume's skill list.


    return matched_skills, missing_skills

# Removed the analyze_resume function from this file as it's redundant
# and should be handled by the resume_parser module.
