import streamlit as st
import pandas as pd
import numpy as np
import sys
import logging
import traceback
from io import BytesIO

# Assuming these plotly imports are used in other parts
import plotly.express as px
import plotly.graph_objects as go
import time # Assuming used somewhere

# --- Import Functions from your Modules ---

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

logger.info("Application started.")

# Import functions from resume_parser.py
# This is now the primary parsing module enabled
try:
    logger.info("Importing from resume_parser...")
    from resume_parser import (
        extract_text_from_pdf,
        extract_text_from_docx,
        analyze_resume as analyze_resume_basic, # Alias to avoid conflict
        # Keep other imports if you use them directly in app.py
        extract_skills # Needed for the basic skill match in Tab 2
    )
    logger.info("Successfully imported from resume_parser.")
    RESUME_PARSER_AVAILABLE = True
except ImportError as e:
    logger.error("❌ Failed to import resume parser module.", exc_info=True)
    st.error("❌ Failed to import resume parser module. Please check logs for details.")
    st.exception(e)
    RESUME_PARSER_AVAILABLE = False
    # Decide if you want to stop the app immediately or show a message
    st.stop()


# TEMPORARILY COMMENTED OUT FOR DIAGNOSIS: Import from semantic_matcher.py
# try:
#     logger.info("Importing from semantic_matcher...")
#     from semantic_matcher import (
#        semantic_match_resume,
#        extract_resume_keywords,
#        # extract_skills_from_text, # Make sure to import if needed
#        # analyze_resume_semantic, # Make sure to import if needed
#        get_skill_matches # This is needed for the basic skill match!
#        # ... other imports from semantic_matcher
#     )
#     logger.info("Successfully imported from semantic_matcher.")
#     SEMANTIC_MATCHER_AVAILABLE = True
# except ImportError as e:
#     logger.warning("Semantic matching module not available. Check logs.", exc_info=True)
#     st.warning("Semantic matching features are not available (module import failed).")
#     SEMANTIC_MATCHER_AVAILABLE = False

# Manually set this flag to False since the import is commented out
SEMANTIC_MATCHER_AVAILABLE = False


st.set_page_config(
    page_title="Smart Job Matcher",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📋 Smart Job Matcher")

st.markdown("""
This app helps you match your resume with job descriptions to find the best fit and improve your application.
""")

tab_names = ["📤 Upload Resume", "🔍 Job Matching", "✨ Resume Enhancement"]
tabs = st.tabs(tab_names)

tab1, tab2, tab3 = tabs

# --- Tab 1: Upload Resume ---
with tab1:
    st.header("Upload Your Resume")
    uploaded_resume = st.file_uploader("Choose your resume file", type=["pdf", "docx"])

    if RESUME_PARSER_AVAILABLE and uploaded_resume is not None:
        try:
            file_details = {
                "Filename": uploaded_resume.name,
                "FileType": uploaded_resume.type,
                "FileSize": f"{uploaded_resume.size / 1024:.2f} KB"
            }
            st.json(file_details)

            resume_text = ""
            file_content_io = uploaded_resume # Streamlit's UploadedFile behaves like BytesIO

            with st.spinner("Reading document..."):
                if uploaded_resume.type == "application/pdf":
                    resume_text = extract_text_from_pdf(file_content_io)
                elif "docx" in uploaded_resume.type or uploaded_resume.name.endswith('.docx'):
                    resume_text = extract_text_from_docx(file_content_io)
                else:
                    st.error("Unsupported file type. Please upload a PDF or DOCX.")
                    # Decide if you want to stop or just show message
                    # st.stop()

            if resume_text and "Error extracting" not in resume_text:
                st.session_state['resume_text'] = resume_text

                with st.spinner("Analyzing your resume..."):
                    resume_analysis_basic = analyze_resume_basic(resume_text)
                    st.session_state['resume_analysis_basic'] = resume_analysis_basic
                    logger.info("Basic resume analysis complete.")

                st.subheader("📊 Extracted Information (Basic Analysis)")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {resume_analysis_basic.get('name', 'N/A')}")
                with col2:
                    st.write(f"**Email:** {resume_analysis_basic.get('email', 'N/A')}")
                    st.write(f"**Phone:** {resume_analysis_basic.get('phone', 'N/A')}")

                if resume_analysis_basic.get('skills'):
                    st.subheader("🔧 Skills:")
                    skill_tags = ''.join([
                        f'<span style="background-color: #f0f2f6; padding: 5px 10px; margin: 2px; border-radius: 15px; display: inline-block;">{skill}</span>'
                        for skill in resume_analysis_basic['skills']
                    ])
                    st.markdown(skill_tags, unsafe_allow_html=True)
                else:
                     st.info("No significant skills extracted by basic method.")

                if resume_analysis_basic.get('education'):
                    st.subheader("🎓 Education:")
                    for edu in resume_analysis_basic['education']:
                        st.markdown(f"- {edu}")
                else:
                     st.info("No education section extracted by basic method.")

                if resume_analysis_basic.get('experience'):
                    st.subheader("💼 Experience Highlights:")
                    for exp in resume_analysis_basic['experience']:
                        st.markdown(f"- {exp}")
                else:
                     st.info("No experience section extracted by basic method.")

            elif "Error extracting" in resume_text:
                 st.error(resume_text)
            else:
                st.error("❌ Failed to extract any text from the resume.")
                logger.warning("Extraction resulted in empty text.")

        except Exception as e:
            logger.error("❌ An unexpected error occurred during resume upload and processing.", exc_info=True)
            st.error("❌ An unexpected error occurred during resume processing. Please check logs.")
            st.exception(e)


# --- Tab 2: Job Matching ---
with tab2:
    st.header("Job Description Analysis")

    if 'resume_text' not in st.session_state or 'resume_analysis_basic' not in st.session_state:
        st.info("👆 Please upload and analyze your resume first (in the 'Upload Resume' tab) to enable job matching.")
    else:
        job_description = st.text_area("Paste the job description here:", height=300, key="job_desc_text_area")

        # Basic Keyword/Skill Matching Section - NOW ONLY USING resume_parser functions
        st.subheader("Keyword/Skill Overlap Match")
        # Only show the button if the basic parser is available
        if RESUME_PARSER_AVAILABLE and st.button("Analyze Keyword Match", key="analyze_keyword_match_button") and job_description:
            with st.spinner("Analyzing job description keywords and matching skills..."):
                try:
                    # Use the skills extracted by the basic method from the resume
                    resume_skills_basic = st.session_state['resume_analysis_basic'].get('skills', [])
                    resume_skills_basic_lower = [s.lower() for s in resume_skills_basic]

                    # --- Perform Keyword Matching using only basic extract_skills ---
                    job_skills_extracted = extract_skills(job_description) # Use extract_skills from resume_parser
                    job_skills_extracted_lower = [s.lower() for s in job_skills_extracted]

                    # Find skills in job_skills_extracted that are present in resume_skills_basic
                    matched_skills_lower = set(job_skills_extracted_lower).intersection(set(resume_skills_basic_lower))

                    # Find skills in job_skills_extracted that are NOT in resume_skills_basic
                    missing_skills_lower = set(job_skills_extracted_lower).difference(set(resume_skills_basic_lower))

                    # Convert back to original casing from the extracted job skills list for display
                    matched_skills = [s for s in job_skills_extracted if s.lower() in matched_skills_lower]
                    missing_skills = [s for s in job_skills_extracted if s.lower() in missing_skills_lower]

                    # Calculate score
                    job_skills_count_lower = len(job_skills_extracted_lower)
                    score = (len(matched_skills_lower) / job_skills_count_lower) * 100 if job_skills_count_lower else 0


                    st.markdown(f"**Match Score:** {score:.1f}%")

                    st.markdown("**✅ Matching Skills:**")
                    if matched_skills:
                        st.markdown(''.join([
                            f'<span style="background-color: #d4edda; color: #155724; padding: 5px 10px; margin: 5px; border-radius: 15px; display: inline-block;">{s}</span>'
                            for s in matched_skills
                        ]), unsafe_allow_html=True)
                    else:
                        st.write("No direct keyword/skill matches found.")

                    st.markdown("**⚠️ Skills to Highlight or Develop:**")
                    if missing_skills:
                        st.markdown(''.join([
                            f'<span style="background-color: #f8d7da; color: #721c24; padding: 5px 10px; margin: 5px; border-radius: 15px; display: inline-block;">{s}</span>'
                            for s in missing_skills
                        ]), unsafe_allow_html=True)
                    else:
                        st.write("You appear to have all the key skills mentioned in the job description (based on keyword extraction).")

                    st.markdown("💡 Recommendations (Keyword Based):")
                    tips = []
                    if score < 50:
                        tips.append("Consider improving skills listed in the job description.")
                    elif score < 75:
                        tips.append("You're a partial match. Emphasize relevant experience and skills.")
                    else:
                        tips.append("You're a strong match based on keywords. Ensure they are prominent on your resume.")

                    if missing_skills:
                         tips.append(f"Consider learning or highlighting experience with these skills: {', '.join(missing_skills[:5])}{'...' if len(missing_skills) > 5 else ''}")

                    for tip in tips:
                         st.markdown(f"- {tip}")


                except Exception as e:
                    logger.error("❌ An unexpected error occurred during keyword match analysis.", exc_info=True)
                    st.error("❌ An unexpected error occurred during keyword match analysis. Please check logs.")
                    st.exception(e)

        # Handle case where button is clicked but no text
        elif RESUME_PARSER_AVAILABLE and st.button("Analyze Keyword Match", key="analyze_keyword_match_button_disabled_placeholder") and not job_description:
             st.warning("Please paste a job description to analyze the match.")


        # TEMPORARILY COMMENTED OUT: Semantic Matching Section
        # if SEMANTIC_MATCHER_AVAILABLE:
        #      st.subheader("🧠 Semantic Relevance Match")
        #      st.info("This feature uses AI to find deeper relevance between your resume text and the job description, beyond just keywords.")

        #      dummy_job_df = pd.DataFrame([{
        #          "Job title": "Pasted Job Description",
        #          "Job description": job_description,
        #          "Requirements": ""
        #      }])

        #      if st.button("Run Semantic Match", key="run_semantic_match_button") and job_description:
        #          with st.spinner("Running advanced semantic analysis..."):
        #              try:
        #                  semantic_results_df = semantic_match_resume(st.session_state['resume_text'], dummy_job_df, top_n=1)

        #                  if not semantic_results_df.empty:
        #                      semantic_score = semantic_results_df.iloc[0]['match_score']
        #                      st.markdown(f"**Semantic Relevance Score:** {semantic_score:.1f}%")
        #                      st.info("Note: Semantic matching provides a broader relevance score based on meaning.")
        #                  else:
        #                      st.warning("Semantic matching did not return results. Check logs for errors.")

        #              except Exception as se:
        #                  logger.error("❌ An unexpected error occurred during semantic matching.", exc_info=True)
        #                  st.error("❌ An unexpected error occurred during semantic matching. Please check logs.")
        #                  st.exception(se)

        #      elif st.button("Run Semantic Match", key="run_semantic_match_button_disabled_placeholder") and not job_description:
        #           st.warning("Please paste a job description to run semantic matching.")

        # else: # Message if semantic matcher is not available at all
        #      st.info("Semantic matching features are currently disabled or unavailable.")


# --- Tab 3: Resume Enhancement ---
with tab3:
    st.header("Resume Enhancement")
    if 'resume_analysis_basic' not in st.session_state:
        st.info("👆 Please upload and analyze your resume first (in the 'Upload Resume' tab) to get enhancement tips.")
    else:
        st.subheader("📝 General Resume Improvement Suggestions")
        tips = [
            "Use strong action verbs to start bullet points in your experience section (e.g., 'Managed', 'Developed', 'Led', 'Implemented').",
            "Quantify your achievements with numbers and metrics whenever possible (e.g., 'Increased sales by 15%', 'Managed a team of 5', 'Reduced costs by $10k').",
            "Tailor your resume, especially the summary/objective and skills sections, to align with the specific keywords and requirements of the jobs you are targeting.",
            "Proofread your resume meticulously for typos and grammatical errors. Consider using a tool like Grammarly.",
            "Ensure consistent formatting, fonts (use standard, professional fonts like Arial, Calibri, Times New Roman), and spacing throughout the document for readability.",
            "Keep your resume concise and focused. Typically 1 page for less than 10 years of experience, and a maximum of 2 pages for more extensive experience.",
            "Consider adding a professional summary or objective statement at the top, tailored to the type of job you seek."
        ]
        for i, t in enumerate(tips):
            st.markdown(f"- {t}")

        st.subheader("🧩 Skills Organization Suggestions")
        # Use the skills extracted by the basic method for this section
        extracted_skills = st.session_state['resume_analysis_basic'].get('skills', [])
        extracted_skills_lower = [s.lower() for s in extracted_skills]

        categories = {
            "Programming Languages": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "go", "rust", "sql"],
            "Web Development": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask"],
            "Data Science / ML": ["tensorflow", "pytorch", "keras", "pandas", "numpy", "scikit-learn", "machine learning", "artificial intelligence", "ai", "ml", "data science"],
            "Cloud Platforms": ["aws", "azure", "gcp", "cloud computing"],
            "DevOps / Tools": ["docker", "kubernetes", "ci/cd", "git", "jira", "agile", "scrum"],
            "Soft Skills": ["leadership", "communication", "teamwork", "problem solving", "critical thinking", "time management", "creativity", "adaptability", "project management", "analytical"]
        }

        if extracted_skills:
            st.markdown("Here's a possible way to group some of the skills found in your resume:")
            categorized_skills_lower = set()

            for cat, keywords in categories.items():
                found_in_category = [
                    s for s in extracted_skills
                    if any(k in s.lower() for k in keywords) and s.lower() not in categorized_skills_lower
                ]

                if found_in_category:
                    st.markdown(f"**{cat}:**")
                    skill_tags = ''.join([
                         f'<span style="background-color: #e9ecef; padding: 3px 8px; margin: 2px; border-radius: 10px; display: inline-block; font-size: 0.9em;">{skill}</span>'
                         for skill in found_in_category
                     ])
                    st.markdown(skill_tags, unsafe_allow_html=True)
                    categorized_skills_lower.update(s.lower() for s in found_in_category)

            uncategorized_skills = [
                s for s in extracted_skills
                if s.lower() not in categorized_skills_lower
            ]

            if uncategorized_skills:
                 st.markdown("**Other Skills:**")
                 skill_tags = ''.join([
                      f'<span style="background-color: #e9ecef; padding: 3px 8px; margin: 2px; border-radius: 10px; display: inline-block; font-size: 0.9em;">{skill}</span>'
                      for skill in uncategorized_skills
                  ])
                 st.markdown(skill_tags, unsafe_allow_html=True)

        else:
            st.info("No skills extracted from your resume to categorize.")

        st.subheader("🤖 Applicant Tracking System (ATS) Optimization Tips")
        st.write("""
        To help your resume pass through Applicant Tracking Systems, consider these points:
        * **Use Standard Section Headings:** Use clear, common titles like "Education", "Experience", "Skills", "Projects", etc.
        * **Include Keywords:** Mirror the language and keywords used in the job description throughout your resume, especially in skills and experience sections.
        * **Simple Formatting:** Avoid complex layouts, tables, text boxes, headers/footers (some systems struggle with these), and heavy graphics. Use bullet points effectively.
        * **Standard Fonts:** Stick to widely recognized fonts (Arial, Calibri, Times New Roman, Georgia, Verdana) and a font size between 10-12pt.
        * **Reverse Chronological Order:** List your experience and education with the most recent items first.
        * **File Type:** Save your resume as a PDF or DOCX. PDF is often preferred for maintaining formatting, but check the application instructions.
        """)


st.markdown("---")
st.markdown("Smart Job Matcher v1.0 – Helping you land your dream job! 🚀")

# --- Sidebar ---
with st.sidebar:
    st.subheader("About")
    st.info("This application analyzes your resume to extract information and compare it with job descriptions.")

    with st.expander("Debug Info"):
        st.write(f"Python: {sys.version}")
        try:
            import spacy
            st.write(f"spaCy: {spacy.__version__}")
            # Check if the spaCy model was successfully loaded in resume_parser
            if 'nlp' in globals() and nlp.meta.get('name') == 'en_core_web_sm':
                 st.write("spaCy model 'en_core_web_sm' loaded.")
            else:
                 st.write("spaCy model 'en_core_web_sm' not loaded or failed.")

        except ImportError:
            st.write("spacy not available")
        except Exception as e:
             st.write(f"Error checking spacy info: {e}")

        try:
             import nltk
             st.write(f"NLTK: {nltk.__version__}")
             try:
                 nltk.data.find('tokenizers/punkt')
                 st.write("NLTK 'punkt' found.")
             except LookupError:
                 st.write("NLTK 'punkt' not found.")
             try:
                 nltk.data.find('corpora/stopwords')
                 st.write("NLTK 'stopwords' found.")
             except LookupError:
                 st.write("NLTK 'stopwords' not found.")
        except ImportError:
             st.write("nltk not available")
        except Exception as e:
             st.write(f"Error checking nltk info: {e}")

        # TEMPORARILY COMMENTED OUT: Sentence-Transformers debug info
        # try:
        #      from sentence_transformers import __version__ as st_version
        #      st.write(f"Sentence-Transformers: {st_version}")
        #      # Check if the model loaded in semantic_matcher (if imported and not commented out)
        #      if SEMANTIC_MATCHER_AVAILABLE and 'model' in dir(semantic_matcher) and isinstance(semantic_matcher.model, SentenceTransformer):
        #           st.write("SentenceTransformer model loaded.")
        #      else:
        #           st.write("SentenceTransformer model not loaded or failed.")
        # except ImportError:
        #      st.write("Sentence-Transformers not available")
        # except Exception as e:
        #      st.write(f"Error checking Sentence-Transformers info: {e}")


# --- Helper function to get logger (redundant but good practice)
# @st.cache_resource
# def get_logger():
#      logger = logging.getLogger(__name__)
#      # Configure logger if needed
#      # if not logger.handlers:
#      #      logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#      return logger
# logger = get_logger()
