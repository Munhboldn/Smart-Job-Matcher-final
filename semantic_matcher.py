# This file is temporarily disabled for diagnosis of the ValueError.
# The code below is commented out.

# import pandas as pd
# import numpy as np
# # from sentence_transformers import SentenceTransformer # Commented out
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import RegexpTokenizer
# import re
# import spacy
# from collections import Counter
# import logging

# # logger = logging.getLogger(__name__)

# # NLTK Data Downloads (redundant if done elsewhere, but harmless)
# # try:
# #     nltk.data.find('tokenizers/punkt')
# # except LookupError:
# #     # logger.info("Downloading NLTK punkt (semantic_matcher)...")
# #     nltk.download('punkt', quiet=True)

# # try:
# #     nltk.data.find('corpora/stopwords')
# # except LookupError:
# #     # logger.info("Downloading NLTK stopwords (semantic_matcher)...")
# #     nltk.download('stopwords', quiet=True)


# # SpaCy Model Loading (Commented out - done in resume_parser.py)
# # try:
# #     # logger.info("Attempting to load spaCy model 'en_core_web_sm' in semantic_matcher.")
# #     nlp_matcher = spacy.load("en_core_web_sm")
# #     # logger.info("Successfully loaded spaCy model 'en_core_web_sm' in semantic_matcher.")
# # except OSError as e:
# #     error_msg = f"FATAL ERROR: Could not load spaCy model 'en_core_web_sm' in semantic_matcher. Ensure it is installed."
# #     # if logger: logger.error(error_msg, exc_info=True)
# #     print(error_msg)
# #     raise # Stop the app if the model isn't available


# # Sentence Transformer Model Loading (Commented out)
# # try:
# #     # logger.info("Loading SentenceTransformer model...")
# #     model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# #     # logger.info("SentenceTransformer model loaded.")
# # except Exception as e:
# #      # if logger: logger.error(f"Error loading SentenceTransformer model: {e}", exc_info=True)
# #      print(f"Error loading SentenceTransformer model: {e}")
# #      # raise # Uncomment if this must stop the app


# # Keyword/Skill Extraction for Matching (Commented out)
# # def extract_resume_keywords(text, top_n=30):
# #     pass # Placeholder

# # def extract_skills_from_text(text):
# #     pass # Placeholder

# # Semantic Matching Function (Commented out)
# # def semantic_match_resume(resume_text, jobs_df, top_n=10):
# #     pass # Placeholder

# # Skill Matching Function (Commented out)
# # def get_skill_matches(resume_skills, job_text):
# #      pass # Placeholder

# # Analyze Resume (Second Version) (Commented out)
# # def analyze_resume_semantic(resume_text):
# #     pass # Placeholder
