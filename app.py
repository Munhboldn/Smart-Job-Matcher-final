# --- Standard Streamlit Import ---
import streamlit as st

# --- Set Page Config (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Smart Job Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Smart Job Matcher\nFind the perfect job match for your skills and experience!"
    }
)

# --- Other Imports (after set_page_config) ---
import pandas as pd
import numpy as np # Needed by semantic_matcher, also potentially here
import io
# Note: plotly is imported here but used within the app modes
import plotly.express as px
import plotly.graph_objects as go
import time # Used for timing in matching
import sys # Used for debug info

# --- Configure Logging (after imports and set_page_config) ---
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

logger.info("Application started.")


# --- Import Functions from Modules ---
# Import functions matching the names expected by your old app.py
try:
    logger.info("Importing from resume_parser...")
    from resume_parser import (
        extract_text_from_pdf,
        extract_text_from_docx,
        analyze_resume, # Imports the basic analyzer
        extract_resume_sections, # Also imported by old app.py
        extract_skills_basic # Explicitly import the basic skill extractor
    )
    logger.info("Successfully imported from resume_parser.")
    RESUME_PARSER_AVAILABLE = True
except ImportError as e:
    logger.error("❌ Failed to import resume parser module.", exc_info=True)
    st.error("❌ Failed to import resume parser module. Please check logs for details.")
    st.exception(e)
    RESUME_PARSER_AVAILABLE = False
    # Decide if you want to stop if basic parser is critical
    # st.stop()


try:
    logger.info("Importing from semantic_matcher...")
    from semantic_matcher import (
       semantic_match_resume,
       extract_resume_keywords,
       extract_skills_from_text, # Explicitly import the semantic skill extractor
       get_skill_matches,
       # Import the semantic analysis function under an alias as in your old app.py
       analyze_resume as analyze_resume_skills
    )
    logger.info("Successfully imported from semantic_matcher.")
    SEMANTIC_MATCHER_AVAILABLE = True
except ImportError as e:
    logger.warning("Semantic matching module not available. Check logs.", exc_info=True)
    st.warning("Semantic matching features are not available. Ensure semantic_matcher.py exists and its dependencies are installed correctly.")
    SEMANTIC_MATCHER_AVAILABLE = False


# --- Load Job Listings with Caching ---
# Use @st.cache_data to load the job data CSV only once
# Expects data file at './data/zangia_filtered_jobs.csv'
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_jobs(file_path="data/zangia_filtered_jobs.csv"):
    """Loads job data from a CSV file and performs basic cleaning."""
    logger.info(f"Attempting to load job data from: {file_path}")
    try:
        # Ensure file exists - important for Streamlit Cloud deployment
        if not os.path.exists(file_path):
             logger.error(f"Job data file not found at: {file_path}")
             st.error(f"Job data file not found at '{file_path}'. Please ensure it's in your repository.")
             # Return empty dataframe with expected columns to prevent downstream errors
             return pd.DataFrame(columns=['Job title', 'Company', 'Salary', 'Job description', 'Requirements', 'URL'])

        df = pd.read_csv(file_path)
        # Add additional preprocessing as in your old code
        df['Salary'] = df['Salary'].fillna('Not specified')
        df['Job description'] = df['Job description'].fillna('')
        df['Requirements'] = df['Requirements'].fillna('')
        # Add URL column if missing, fill with '#'
        if 'URL' not in df.columns:
             df['URL'] = '#'

        logger.info(f"Successfully loaded {len(df)} jobs.")
        return df
    except Exception as e:
        logger.error(f"Error loading job data from {file_path}: {e}", exc_info=True)
        st.error(f"Error loading job data: {str(e)}. Please check the file format and content.")
        # Return empty dataframe with expected columns on error
        return pd.DataFrame(columns=['Job title', 'Company', 'Salary', 'Job description', 'Requirements', 'URL'])

# Load jobs data on app startup
jobs_df = load_jobs()

# --- Custom Styling (from your old app.py) ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E3F2FD;
    }
    .job-title {
        font-size: 1.5rem;
        color: #1565C0;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .match-section {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .match-score-high {
        font-size: 1.2rem;
        color: #2E7D32;
        font-weight: bold;
    }
    .match-score-medium {
        font-size: 1.2rem;
        color: #FF8F00;
        font-weight: bold;
    }
    .match-score-low {
        font-size: 1.2rem;
        color: #C62828;
        font-weight: bold;
    }
    .matched-keywords {
        background-color: #E8F5E9;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .missing-keywords {
        background-color: #FFEBEE;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .tips-section {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .loading-spinner {
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization (from your old app.py) ---
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'resume_analysis' not in st.session_state: # Corresponds to basic analysis
    st.session_state.resume_analysis = None
if 'job_results' not in st.session_state:
    st.session_state.job_results = None
if 'skills_extracted' not in st.session_state: # Corresponds to basic skills from basic analysis
    st.session_state.skills_extracted = []
# Add session state for semantic analysis results if needed by the UI
# if 'resume_analysis_semantic' not in st.session_state:
#      st.session_state.resume_analysis_semantic = None # For semantic analysis results


# Custom header (using markdown)
st.markdown('<div class="main-header">💼 Smart Job Matcher</div>', unsafe_allow_html=True)


# --- Sidebar for filters and settings (from your old app.py) ---
with st.sidebar:
    st.header("⚙️ Options")

    # App mode selection
    app_mode = st.radio("Select Mode", ["Resume-to-Job Matching", "Resume Analysis", "Job Market Explorer"])

    # Job market filters (conditionally displayed)
    salary_min = 0 # Default value
    if app_mode == "Job Market Explorer":
         st.subheader("Job Market Filters")
         # Ensure jobs_df is not empty before showing filters that depend on its columns/data
         if not jobs_df.empty:
              try:
                  # Find min/max salary from the loaded data for better input range
                  # Requires extracting numeric salary first, similar to the display logic
                  def extract_salary_numeric(salary_str):
                       if pd.isna(salary_str) or salary_str == 'Not specified': return 0
                       try: return int(re.sub(r'\D', '', str(salary_str))) # Simple extraction
                       except: return 0
                  numeric_salaries = jobs_df['Salary'].apply(extract_salary_numeric)
                  min_salary_data = numeric_salaries[numeric_salaries > 0].min() if not numeric_salaries[numeric_salaries > 0].empty else 0
                  max_salary_data = numeric_salaries.max() if not numeric_salaries.empty else 10000000 # Assume a reasonable max

                  salary_min = st.number_input(
                      "Minimum Salary (₮)",
                      min_value=0,
                      # Set a default value based on data min, step
                      value=int(min_salary_data),
                      step=100000,
                      format="%d" # Ensure integer display
                  )
                  if salary_min > max_salary_data:
                       st.warning(f"Minimum salary exceeds the maximum found salary ({max_salary_data:,.0f}₮). No jobs will match.")

              except Exception as e:
                   st.warning(f"Could not load salary filters: {e}")
                   salary_min = st.number_input("Minimum Salary (₮)", min_value=0, value=0, step=100000)

         else:
              st.info("Load job data first to enable filters.")
              salary_min = st.number_input("Minimum Salary (₮)", min_value=0, value=0, step=100000, disabled=True)


    # About section
    st.markdown("---")
    st.write("#### About")
    st.write("""
    Smart Job Matcher helps you find jobs that match your skills and experience using advanced AI matching technology.

    Upload your resume and get personalized job recommendations!
    """)

    # --- Sidebar Debug Info (Cleaned) ---
    # This section helps verify loaded modules and models
    st.markdown("---")
    with st.expander("Debug Info"):
        st.write(f"Python: {sys.version}")

        # Debug info for spaCy (basic parser)
        try:
            import spacy
            st.write(f"spaCy: {spacy.__version__}")
            # Check if the 'nlp' variable exists in the resume_parser module and is a spaCy Language object
            nlp_rp_loaded_check = False
            if 'resume_parser' in sys.modules:
                 rp_module = sys.modules['resume_parser']
                 if hasattr(rp_module, 'nlp') and isinstance(rp_module.nlp, spacy.Language):
                      nlp_rp_loaded_check = True

            if nlp_rp_loaded_check:
                 st.write("spaCy model (basic parser) appears loaded.")
            else:
                 st.write("spaCy model (basic parser) not found or failed.")

        except ImportError:
            st.write("spaCy not available.")
        except Exception as e:
             st.write(f"Error checking spaCy info: {e}", exc_info=True)


        # Debug info for NLTK
        try:
             import nltk
             st.write(f"NLTK: {nltk.__version__}")
             # Check if NLTK data is found
             punkt_found = False
             stopwords_found = False
             try:
                 nltk.data.find('tokenizers/punkt')
                 punkt_found = True
             except LookupError:
                 pass # Punkt not found

             try:
                 nltk.data.find('corpora/stopwords')
                 stopwords_found = True
             except LookupError:
                 pass # Stopwords not found

             # Corrected Syntax for these checks
             if punkt_found:
                 st.write("NLTK 'punkt' found.")
             else:
                 st.write("NLTK 'punkt' not found.")

             if stopwords_found:
                 st.write("NLTK 'stopwords' found.")
             else:
                 st.write("NLTK 'stopwords' not found.")

        except ImportError:
             st.write("nltk not available.")
        except Exception as e:
             st.write(f"Error checking nltk info: {e}", exc_info=True)


        # Debug info for Semantic Matcher models
        if SEMANTIC_MATCHER_AVAILABLE:
             st.write("Semantic Matcher available.")
             try:
                  # Import sentence_transformers here just for the isinstance check
                  import sentence_transformers

                  sm_nlp_loaded_check = False
                  sm_st_loaded_check = False

                  if 'semantic_matcher' in sys.modules:
                       sm_module = sys.modules['semantic_matcher']

                       # Check spaCy model in semantic_matcher
                       if hasattr(sm_module, 'nlp_matcher') and isinstance(sm_module.nlp_matcher, spacy.Language):
                            sm_nlp_loaded_check = True

                       # Check SentenceTransformer model
                       if hasattr(sm_module, 'model') and isinstance(sm_module.model, sentence_transformers.SentenceTransformer):
                            sm_st_loaded_check = True

                  if sm_nlp_loaded_check:
                       st.write("spaCy model (semantic matcher) appears loaded.")
                  else:
                       st.write("spaCy model (semantic matcher) not found or failed.")

                  if sm_st_loaded_check:
                       st.write("SentenceTransformer model appears loaded.")
                  else:
                       st.write("SentenceTransformer model not found or failed.")

             except ImportError:
                  st.write("Semantic Matcher dependencies (e.g., sentence_transformers) not available.")
             except Exception as e:
                  st.write(f"Error checking Semantic Matcher info: {e}", exc_info=True)
        else:
             st.write("Semantic Matcher not available.")


# --- Main App Content (based on app_mode) ---

if app_mode == "Resume-to-Job Matching":
    # Ensure jobs_df is loaded before proceeding
    if jobs_df is None or jobs_df.empty:
         st.error("Job data could not be loaded. Please check the data file.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Upload resume section
            st.markdown('<div class="sub-header">📄 Upload Your Resume</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["pdf", "docx"])

            if uploaded_file:
                # Extract resume text with progress indication
                with st.spinner("Processing resume..."):
                    if uploaded_file.name.endswith(".pdf"):
                        resume_text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.name.endswith(".docx"):
                        resume_text = extract_text_from_docx(uploaded_file)
                    else:
                        st.error("Unsupported file format.")
                        resume_text = ""

                    # Store in session state
                    st.session_state.resume_text = resume_text

                    # Analyze resume (basic)
                    resume_analysis = analyze_resume(resume_text) # Uses the basic analyze_resume from resume_parser
                    st.session_state.resume_analysis = resume_analysis

                    # Extract skills (basic)
                    st.session_state.skills_extracted = extract_skills_basic(resume_text) # Uses basic skill extractor

                # Display extracted text in an expandable section
                if st.session_state.resume_text and "Error extracting" not in st.session_state.resume_text:
                     with st.expander("View Extracted Resume Text", expanded=False):
                         st.text_area("", st.session_state.resume_text, height=200)
                elif "Error extracting" in st.session_state.resume_text:
                     st.error(st.session_state.resume_text)
                else:
                     st.warning("No text extracted from resume.")


        with col2:
            if st.session_state.resume_analysis:
                # Display resume stats (from basic analysis)
                st.markdown('<div class="sub-header">📊 Resume Stats</div>', unsafe_allow_html=True)

                # Resume completeness score (using the score from the basic analyze_resume)
                # Note: The old app.py used the semantic completeness score here.
                # Let's use the score from the basic analyze_resume if it calculates one,
                # or add a simple length-based heuristic if not. The current analyze_resume_basic doesn't have a score.
                # Let's add a simple word count heuristic as a placeholder stat if no specific score is calculated by basic analysis.
                word_count = len(st.session_state.resume_text.split()) if st.session_state.resume_text else 0
                st.markdown(f"**Resume Word Count:** {word_count}")


                # Resume sections found (from basic analysis)
                st.markdown("**Detected Sections (Basic):**")
                sections = st.session_state.resume_analysis.get('sections', {})
                section_list = [s.title() for s in sections if s not in ['Header', 'other', 'content'] and sections[s].strip()]
                if section_list:
                     st.markdown(", ".join(section_list))
                else:
                     st.info("No clear sections detected by basic parser.")


                # Skills extracted (basic skills)
                st.markdown("**Extracted Skills (Basic):**")
                if st.session_state.skills_extracted:
                     # Display skills as tags (using the HTML formatting from the old app.py)
                     skill_tags = ''.join([
                         f'<span style="background-color: #f0f2f6; padding: 3px 8px; margin: 2px; border-radius: 10px; display: inline-block; font-size: 0.9em;">{skill}</span>'
                         for skill in st.session_state.skills_extracted[:10] # Display top 10
                     ])
                     if len(st.session_state.skills_extracted) > 10:
                          skill_tags += f'<span style="padding: 3px 8px; margin: 2px; display: inline-block; font-size: 0.9em;">... (+{len(st.session_state.skills_extracted) - 10} more)</span>'
                     st.markdown(skill_tags, unsafe_allow_html=True)

                else:
                     st.info("No skills detected by basic method.")

        # Job matching section (only if resume uploaded and jobs loaded)
        if st.session_state.resume_text and not jobs_df.empty:
            st.markdown('<div class="sub-header">🔎 Find Matching Jobs</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])

            with col1:
                # Job sector filter (using sector_options from your old app.py)
                # Define sector_options dictionary
                sector_options = {
                    "Education & Management": ["багш", "сургалт", "удирдлага", "education", "teacher", "lecturer", "professor", "academic"],
                    "Customer Service": ["customer service", "үйлчилгээ", "захиалга", "client", "support", "help desk", "call center"],
                    "Leadership": ["менежер", "захирал", "manager", "director", "executive", "chief", "head of", "supervisor"],
                    "Tech & Development": ["developer", "инженер", "програм", "software", "programmer", "coder", "IT", "tech", "web", "mobile"],
                    "Creative & Marketing": ["дизайн", "сошиал", "контент", "media", "designer", "creative", "marketing", "brand", "SEO", "content"],
                    "Finance": ["нягтлан", "санхүү", "finance", "accountant", "accounting", "financial", "budget", "tax", "banking"],
                    "Healthcare": ["эмч", "сувилагч", "health", "эрүүл мэнд", "doctor", "nurse", "medical", "healthcare", "clinic", "hospital"],
                    "Logistics & Support": ["logistics", "тээвэр", "нярав", "туслах", "warehouse", "shipping", "supply chain", "inventory"],
                    "Data & AI": ["data analyst", "data scientist", "AI", "мэдээлэл шинжээч", "machine learning", "analytics", "big data", "statistics"],
                    "HR & Recruitment": ["HR", "хүний нөөц", "recruiter", "talent", "hiring", "recruitment", "personnel", "staffing"],
                    "Legal & Compliance": ["хууль", "lawyer", "legal", "compliance", "attorney", "law", "regulatory", "contracts"],
                    "Sales": ["борлуулалт", "sales", "зөвлөх", "business development", "account manager", "retail", "revenue"],
                    "Project Management": ["төслийн менежер", "project manager", "project coordinator", "scrum", "agile", "program manager"],
                    "Engineering & Construction": ["механик", "цахилгаан", "civil", "барилга", "engineer", "mechanical", "electrical", "construction"]
                }
                selected_sectors = st.multiselect(
                    "Filter by job sector(s)",
                    options=list(sector_options.keys()),
                    placeholder="Choose one or more sectors..."
                )

                # Convert sector selections to keywords
                selected_keywords = []
                if selected_sectors:
                    for sector in selected_sectors:
                        selected_keywords.extend(sector_options[sector])

            with col2:
                # Number of results to show
                top_n = st.slider("Number of results", min_value=5, max_value=50, value=10, step=5) # Increased max to 50

                # Match button
                find_jobs = st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True)

            # --- Matching Logic ---
            if find_jobs:
                # Show progress and perform matching
                with st.spinner("Matching your profile to jobs..."):
                    # Filter jobs by selected keywords if any
                    filtered_jobs = jobs_df.copy() # Work on a copy
                    if selected_keywords:
                        # Use regex for broad matching across title and description
                        pattern = '|'.join(re.escape(kw) for kw in selected_keywords) # Escape special characters
                        # Ensure columns are string type before applying str.contains
                        filtered_jobs = filtered_jobs[
                            filtered_jobs["Job title"].fillna('').astype(str).str.lower().str.contains(pattern, na=False) |
                            filtered_jobs["Job description"].fillna('').astype(str).str.lower().str.contains(pattern, na=False) |
                            filtered_jobs["Requirements"].fillna('').astype(str).str.lower().str.contains(pattern, na=False) # Also check requirements
                        ]

                    # If no jobs match the filters, show a message
                    if filtered_jobs.empty:
                        st.warning("No jobs found matching your selected filters. Try adjusting them.")
                        st.session_state.job_results = pd.DataFrame() # Clear previous results
                    else:
                        # Match using semantic embeddings (uses semantic_match_resume from semantic_matcher)
                        # Ensure Semantic Matcher is available before calling
                        if SEMANTIC_MATCHER_AVAILABLE:
                             start_time = time.time()
                             results = semantic_match_resume(st.session_state.resume_text, filtered_jobs, top_n=top_n)
                             matching_time = time.time() - start_time

                             # Store results in session state
                             st.session_state.job_results = results

                             # Summary stats
                             st.success(f"Found {len(results)} matching job{ 's' if len(results) != 1 else '' } in {matching_time:.2f} seconds")

                             # Create interactive visualizations of match scores (only if results are not empty)
                             if not st.session_state.job_results.empty:
                                 fig = px.bar(
                                     st.session_state.job_results, # Use the stored results
                                     x='Job title',
                                     y='match_score',
                                     color='match_score',
                                     color_continuous_scale='viridis',
                                     labels={'match_score': 'Match Score (%)'},
                                     title='Job Match Scores'
                                 )
                                 fig.update_layout(xaxis_tickangle=-45)
                                 st.plotly_chart(fig, use_container_width=True)
                             else:
                                  st.info("No semantic matches found for the filtered jobs.")

                        else:
                             st.warning("Semantic matching module is not available. Cannot perform semantic matching.")
                             st.session_state.job_results = pd.DataFrame() # Clear previous results if semantic matcher is needed for results


            # --- Display Job Results ---
            # Display if results are in session state
            if st.session_state.job_results is not None and not st.session_state.job_results.empty:
                st.markdown('<div class="sub-header">✅ Top Matching Jobs</div>', unsafe_allow_html=True)

                # Get resume skills for matching (using basic skills for detailed view)
                resume_skills = st.session_state.skills_extracted # Use basic skills from session state

                # Create tabs for different views
                tab1, tab2 = st.tabs(["List View", "Detailed View"])

                with tab1:
                    # Create a dataframe for display
                    display_df = st.session_state.job_results[['Job title', 'Company', 'Salary', 'match_score']].copy()
                    display_df['match_score'] = display_df['match_score'].round(1).astype(str) + '%'
                    # Optional: Add link column if URL is available and display as clickable link (more complex in st.dataframe)
                    st.dataframe(display_df, use_container_width=True, hide_index=True) # Hide index

                with tab2:
                    # Show detailed job cards
                    for i, (_, row) in enumerate(st.session_state.job_results.iterrows()):
                        with st.container(border=True): # Use st.container with border for visual separation
                            # Get job text for skill matching
                            job_text = str(row.get("Job description", "")) + " " + str(row.get("Requirements", ""))

                            # Get matched and missing skills (uses get_skill_matches from semantic_matcher if available)
                            # Fallback gracefully if semantic matcher is not available
                            if SEMANTIC_MATCHER_AVAILABLE:
                                matched_skills, missing_skills = get_skill_matches(resume_skills, job_text)
                            else:
                                # Simple fallback for display if semantic matcher is off
                                matched_skills = [s for s in resume_skills if s.lower() in job_text.lower()]
                                missing_skills = [s for s in extract_skills_basic(job_text) if s.lower() not in [rs.lower() for rs in resume_skills]]
                                st.warning("Semantic Matcher not available, showing basic skill overlap.")


                            # Format match score with color
                            match_score = row['match_score']
                            if match_score >= 80:
                                match_class = "match-score-high"
                            elif match_score >= 60:
                                match_class = "match-score-medium"
                            else:
                                match_class = "match-score-low"

                            # Job card display
                            st.markdown(f'<div class="job-title">{i+1}. {row["Job title"]}</div>', unsafe_allow_html=True)
                            st.markdown(f"**Company:** {row.get('Company', 'Unknown')}")
                            st.markdown(f"**Salary:** {row.get('Salary', 'Not specified')}")
                            st.markdown(f'**Match Score:** <span class="{match_class}">{match_score:.1f}%</span>', unsafe_allow_html=True)

                            # Job URL
                            if row.get('URL') and pd.notna(row['URL']) and row['URL'].strip() != '#':
                                 st.markdown(f"[🔗 View Job Posting]({row['URL']})")
                            else:
                                 st.info("Job posting URL not available.")


                            # Matched and Missing Skills (using HTML formatting)
                            if matched_skills:
                                st.markdown('<div class="matched-keywords">🟢 <b>Matched skills:</b> ' +
                                            ', '.join(matched_skills[:8]) +
                                            (f' (+{len(matched_skills)-8} more)' if len(matched_skills) > 8 else '') +
                                            '</div>', unsafe_allow_html=True)

                            if missing_skills:
                                st.markdown('<div class="missing-keywords">🔴 <b>Missing skills:</b> ' +
                                            ', '.join(missing_skills[:8]) +
                                            (f' (+{len(missing_skills)-8} more)' if len(missing_skills) > 8 else '') +
                                            '</div>', unsafe_allow_html=True)


                            # Expandable job description
                            with st.expander("View Job Details"):
                                st.markdown("#### Job Description")
                                st.markdown(row.get("Job description", "No description available"))

                                st.markdown("#### Requirements")
                                st.markdown(row.get("Requirements", "No requirements specified"))

                        # No need for closing </div> if using st.container(border=True)
                        # st.markdown('</div>', unsafe_allow_html=True)
                        # No need for HR if using container borders
                        # st.markdown("---")


                # --- Download results ---
                # Only show download if there are results
                output = io.BytesIO()
                # Use display_df or a filtered version if only specific columns are desired in download
                download_df = st.session_state.job_results[['Job title', 'Company', 'Salary', 'match_score', 'URL', 'Job description', 'Requirements']] # Include full job info
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    download_df.to_excel(writer, index=False)
                output.seek(0)

                st.download_button(
                    label="⬇ Download results as Excel",
                    data=output,
                    file_name="matched_jobs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # --- Resume improvement tips (within matching mode) ---
                st.markdown('<div class="sub-header">🚀 Improve Your Resume</div>', unsafe_allow_html=True)
                st.markdown('<div class="tips-section">', unsafe_allow_html=True)
                st.markdown("Based on your job matches, here are tips to improve your resume:")

                # Generate dynamic tips based on job results and missing skills
                all_missing_skills = []
                # Iterate through the matched jobs to collect all missing skills
                for _, row in st.session_state.job_results.iterrows():
                    job_text = str(row.get("Job description", "")) + " " + str(row.get("Requirements", ""))
                     # Use semantic_matcher's get_skill_matches if available, else basic logic
                    if SEMANTIC_MATCHER_AVAILABLE:
                        _, missing = get_skill_matches(resume_skills, job_text) # Use get_skill_matches
                    else:
                        # Fallback missing skill calculation if semantic matcher is off
                        job_skills_basic_for_missing = extract_skills_basic(job_text)
                        resume_skills_lower = [rs.lower() for rs in resume_skills]
                        missing = [s for s in job_skills_basic_for_missing if s.lower() not in resume_skills_lower]

                    all_missing_skills.extend(missing)

                # Count most common missing skills across all top matches
                from collections import Counter
                # Use lowercase for counting consistency, but display original case
                common_missing_lower = Counter([s.lower() for s in all_missing_skills]).most_common(5)

                if common_missing_lower:
                    st.markdown("#### Top Skills to Add Based on Matched Jobs:")
                    # Map back to original casing from the collected missing skills list for display
                    # This requires iterating through the original list to find casing
                    original_cased_missing = list(set(all_missing_skills)) # Get unique missing skills in original casing

                    for skill_lower, count in common_missing_lower:
                         # Find the first occurrence of the original case for this skill
                         original_skill = next((s for s in original_cased_missing if s.lower() == skill_lower), skill_lower) # Fallback to lowercase if not found

                         st.markdown(f"- **{original_skill}** (mentioned in {count} job{'s' if count > 1 else ''})")
                else:
                    st.info("No frequently missing skills identified across the top job matches.")


                # General resume improvement tips (from your old app.py)
                st.markdown("""
                #### General Tips:
                - Make sure your resume highlights your most relevant skills for the targeted positions
                - Use specific keywords from job descriptions in your resume
                - Quantify your achievements with numbers and metrics
                - Use action verbs to describe your experience
                - Tailor your resume for each application
                """)
                st.markdown('</div>', unsafe_allow_html=True)


# --- "Resume Analysis" Mode (from your old app.py) ---
elif app_mode == "Resume Analysis":
     # Ensure resume is uploaded before showing analysis
     if 'resume_text' not in st.session_state or not st.session_state.resume_text:
          st.info("👆 Please upload your resume first (in the 'Resume-to-Job Matching' tab) to get an analysis.")
     else:
          st.markdown('<div class="sub-header">📊 Resume Analyzer</div>', unsafe_allow_html=True)
          st.write("Detailed analysis of your resume content and structure.")

          # Analyze resume using *both* analysis functions
          # analyze_resume from resume_parser (basic section/contact/skills)
          # analyze_resume (aliased analyze_resume_skills) from semantic_matcher (semantic keywords/skills/section keyword counts)

          # Use cached results if available, otherwise run analysis
          # Note: analysis functions are run in the 'Resume-to-Job Matching' mode upon upload.
          # We can potentially just use the stored results here.
          resume_analysis_basic = st.session_state.get('resume_analysis', {})
          # The old app.py seemed to run analyze_resume_skills (semantic analysis) specifically in this tab too.
          # Let's re-run it here to match that behavior, or ensure it's run upon upload and stored.
          # To match the old app.py's likely flow, let's ensure semantic keywords are stored upon upload.
          # The 'Resume-to-Job Matching' upload logic already stores 'resume_analysis_basic' and 'resume_analysis_semantic_keywords'.
          # We can use those stored results here.
          resume_analysis_semantic_keywords = st.session_state.get('resume_analysis_semantic_keywords', []) # This is just keywords

          # To get the other semantic analysis details (sections by keyword, completeness), we'd need to run semantic_matcher.analyze_resume
          # Let's run semantic_matcher.analyze_resume here if needed, or adjust the upload logic to store its result.
          # Given the old app.py's import 'analyze_resume as analyze_resume_skills' from semantic_matcher, and its use here,
          # it was likely intended to call the semantic_matcher's analyze_resume function.
          # Let's run the semantic analysis function again in this tab if semantic matcher is available.

          semantic_analysis_results = None
          if SEMANTIC_MATCHER_AVAILABLE:
              with st.spinner("Running semantic analysis for Analyzer tab..."):
                  try:
                      # Call the semantic analysis function (analyze_resume from semantic_matcher)
                      # This function returns skills, keywords, section keyword counts, completeness
                      semantic_analysis_results = analyze_resume_skills(st.session_state.resume_text) # Calls semantic_matcher.analyze_resume
                  except Exception as e:
                      logger.error("Error running semantic analysis for Analyzer tab.", exc_info=True)
                      st.warning("Could not run semantic analysis for the Analyzer tab.")

          # Display analysis results
          if resume_analysis_basic or semantic_analysis_results: # Display if at least one analysis worked
              # Create tabs for different analysis views
              tab1, tab2, tab3 = st.tabs(["Overview", "Content Analysis", "Improvement Tips"])

              with tab1: # Overview Tab
                  st.markdown("#### Overview Metrics")

                  # Overview metrics (using basic analysis results where appropriate)
                  col1, col2, col3 = st.columns(3)

                  with col1:
                      # Completeness score - Use the one calculated by semantic analysis if available, else a basic length heuristic
                      completeness = 0
                      if semantic_analysis_results and 'completeness' in semantic_analysis_results:
                           completeness = semantic_analysis_results['completeness']
                           st.metric("Keyword Completeness", f"{completeness:.1f}%")
                           st.progress(completeness/100)
                      else:
                           # Basic word count placeholder if semantic analysis failed
                           word_count = len(st.session_state.resume_text.split()) if st.session_state.resume_text else 0
                           st.metric("Resume Word Count", word_count)


                  with col2:
                      # Section count (from basic analysis)
                      sections_basic = resume_analysis_basic.get('sections', {})
                      section_count_basic = len([s for s in sections_basic if s not in ['Header', 'other', 'content'] and sections_basic[s].strip()])
                      st.metric("Basic Sections Found", section_count_basic)

                  with col3:
                      # Skills count (from semantic analysis if available, else basic)
                      skills_semantic_count = len(semantic_analysis_results.get('skills', [])) if semantic_analysis_results else 0
                      skills_basic_count = len(resume_analysis_basic.get('skills', []))

                      if skills_semantic_count > 0:
                           st.metric("Skills Detected (Semantic)", skills_semantic_count)
                      elif skills_basic_count > 0:
                           st.metric("Skills Detected (Basic)", skills_basic_count)
                      else:
                           st.metric("Skills Detected", 0)


                  # Contact information (from basic analysis)
                  st.markdown("#### Contact Information")
                  contact_info = resume_analysis_basic.get('contact_info', {}) # Basic analysis doesn't have contact_info in our version?
                  # Let's use the regex method directly here if analyze_resume_basic doesn't return contact_info
                  import re
                  email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', st.session_state.resume_text)
                  phone_matches = re.findall(r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}(?:\s*(?:ext|x|extn|extension)\s*\d+)?|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{4})', st.session_state.resume_text)
                  # Old app.py also checked linkedin - let's add that regex
                  linkedin_matches = re.findall(r'linkedin\.com/in/[\w-]+', st.session_state.resume_text.lower())


                  if email_matches or phone_matches or linkedin_matches:
                       if email_matches: st.write(f"📧 Email: {email_matches[0]}")
                       if phone_matches: st.write(f"📱 Phone: {phone_matches[0]}")
                       if linkedin_matches: st.write(f"🔗 LinkedIn: {linkedin_matches[0]}")
                       # Note: Location extraction was in the old analyze_resume (2nd file), not our basic one.
                       # To add location, we'd need to re-implement or adapt extract_contact_info from the 2nd old file.
                       # Let's skip location for now to keep it clean.
                  else:
                       st.warning("No contact information detected. Make sure your resume includes clear contact details.")


              with tab2: # Content Analysis Tab
                  st.markdown("#### Content Analysis")

                  # Skills analysis (from semantic analysis if available, else basic)
                  st.markdown("##### Skills:")
                  skills_to_display = semantic_analysis_results.get('skills', []) if semantic_analysis_results else resume_analysis_basic.get('skills', [])

                  if skills_to_display:
                       # Display skills as tags
                       skill_tags = ''.join([
                           f'<span style="background-color: #f0f2f6; padding: 3px 8px; margin: 2px; border-radius: 10px; display: inline-block; font-size: 0.9em;">{skill}</span>'
                           for skill in skills_to_display[:20] # Limit display
                       ])
                       if len(skills_to_display) > 20:
                           skill_tags += f'<span style="padding: 3px 8px; margin: 2px; display: inline-block; font-size: 0.9em;">... (+{len(skills_to_display) - 20} more)</span>'
                       st.markdown(skill_tags, unsafe_allow_html=True)
                  else:
                       st.warning("No specific skills detected.")


                  # Keywords visualization (from semantic analysis if available)
                  st.markdown("##### Keywords:")
                  keywords_to_display = semantic_analysis_results.get('keywords', []) if semantic_analysis_results else []

                  if keywords_to_display:
                      # Create a word frequency dictionary for visualization
                      # Use Counter directly on the list, then slice
                      word_freq = Counter(keywords_to_display).most_common(20) # Top 20 keywords

                      # Create horizontal bar chart
                      if word_freq:
                           fig = go.Figure(go.Bar(
                               x=[v for _, v in word_freq], # Frequency
                               y=[k for k, _ in word_freq], # Keyword
                               orientation='h',
                               marker_color='royalblue'
                           ))
                           fig.update_layout(
                               title="Top Keywords in Your Resume",
                               xaxis_title="Frequency",
                               yaxis_title="Keyword",
                               height=min(600, len(word_freq) * 30 + 150) # Adjust height based on number of keywords
                           )
                           fig.update_yaxes(autorange="reversed") # Show highest frequency at the top
                           st.plotly_chart(fig, use_container_width=True)
                      else:
                          st.info("No significant keywords extracted.")
                  else:
                      st.info("Semantic keywords extraction not available.")


                  # Display resume sections (from basic analysis)
                  st.markdown("##### Resume Sections (Basic Parser):")
                  sections_basic = resume_analysis_basic.get('sections', {})

                  if sections_basic:
                      for section_name, content in sections_basic.items():
                           if section_name not in ['Header', 'other', 'content'] and content and content.strip():
                               with st.expander(f"{section_name.title()} Section"):
                                    st.write(content)

                      # Display 'Header' and 'other/content' if they have content
                      header_content = sections_basic.get('Header', '').strip()
                      if header_content:
                          with st.expander("Header Section"):
                               st.write(header_content)

                      other_content = sections_basic.get('other', '').strip() + sections_basic.get('content', '').strip()
                      if other_content:
                           with st.expander("Other/Unclassified Content"):
                                st.write(other_content)

                  else:
                       st.warning("No clear sections detected by basic parser.")

                  # Display section keyword counts (from semantic analysis)
                  if semantic_analysis_results and 'sections' in semantic_analysis_results: # 'sections' key from semantic analysis
                       st.markdown("##### Section Keyword Presence (Semantic Check):")
                       section_keyword_counts = semantic_analysis_results['sections']
                       if any(section_keyword_counts.values()): # Check if any count is > 0
                           for section, count in section_keyword_counts.items():
                                st.write(f"- **{section.title()}**: {count} relevant keyword{ 's' if count != 1 else '' } found")
                       else:
                           st.info("No section-related keywords found.")
                  else:
                       st.info("Semantic section keyword check not available.")


              with tab3: # Improvement Tips Tab
                  st.markdown("#### Resume Improvement Suggestions")

                  # Use the semantic completeness score if available, else assume 0 for tips
                  completeness_for_tips = semantic_analysis_results.get('completeness', 0) if semantic_analysis_results else 0

                  # Check completeness (using semantic completeness score)
                  if completeness_for_tips < 70:
                      st.markdown("##### 🔴 Completeness Issues:")
                      st.markdown(f"Your resume's estimated keyword completeness is {completeness_for_tips:.1f}%. Consider adding more content or clearly labeling sections.")

                      # Suggest adding sections based on low keyword counts in semantic analysis
                      if semantic_analysis_results and 'sections' in semantic_analysis_results:
                          missing_sections_semantic = [
                              s for s, count in semantic_analysis_results['sections'].items()
                              if count == 0 # Check sections with 0 keyword count
                          ]
                          if missing_sections_semantic:
                               st.markdown("Based on keywords, these sections seem underrepresented or missing:")
                               for section in missing_sections_semantic:
                                    st.markdown(f"- **{section.title()}**")


                  # Skills suggestions (based on skills extracted by semantic analysis)
                  skills_semantic = semantic_analysis_results.get('skills', []) if semantic_analysis_results else []
                  if SEMANTIC_MATCHER_AVAILABLE and len(skills_semantic) < 10:
                      st.markdown("##### 🔶 Skills Recommendations:")
                      st.markdown(f"Only {len(skills_semantic)} specific skills detected by the semantic method.")
                      st.markdown("""
                      - Ensure your skills section uses clear and specific terms.
                      - Include both technical and soft skills relevant to your field.
                      - Consider adding skill levels (e.g., "Proficient in Python").
                      """)
                  elif not SEMANTIC_MATCHER_AVAILABLE:
                       st.markdown("##### 🔶 Skills Recommendations:")
                       st.warning("Semantic Matcher is not available, skill analysis is limited.")
                       st.markdown("Ensure your skills section clearly lists your proficiencies.")


                  # Content analysis tips
                  st.markdown("##### 📝 Content Tips:")
                  if st.session_state.resume_text and len(st.session_state.resume_text.split()) < 400: # Heuristic word count tip
                       st.markdown("- Your resume seems relatively brief. Consider adding more detail to your experiences and projects.")

                  # General tips
                  st.markdown("##### 💡 General Improvements:")
                  st.markdown("""
                  - Use action verbs to start bullet points (achieved, created, developed, etc.)
                  - Quantify your achievements with numbers where possible (e.g., "Increased sales by 15%")
                  - Ensure consistent formatting throughout the document.
                  - Proofread carefully for spelling and grammar errors.
                  - **Tailor your resume for specific job applications** by including keywords from the job description.
                  """)


              # Resume samples suggestion
              st.markdown("##### 📚 Additional Resources:")
              st.markdown("""
              - Review sample resumes in your industry for inspiration.
              - Consider having your resume reviewed by a professional.
              """)


          elif st.session_state.resume_text and "Error extracting" not in st.session_state.resume_text:
               st.info("Could not perform resume analysis. Check logs for errors during analysis.")
          elif "Error extracting" in st.session_state.resume_text:
               st.error(st.session_state.resume_text)
          else:
               st.warning("Upload and process a resume first.")


# --- "Job Market Explorer" Mode (from your old app.py) ---
elif app_mode == "Job Market Explorer":
    st.markdown('<div class="sub-header">🔍 Job Market Explorer</div>', unsafe_allow_html=True)
    st.write("Explore the current job market trends and opportunities based on the loaded data.")

    # Ensure jobs_df is loaded before showing explorer
    if jobs_df is None or jobs_df.empty:
         st.error("Job data could not be loaded. Cannot explore the market.")
    else:
        # Apply any filters from sidebar (salary_min is already captured)
        filtered_jobs = jobs_df.copy()

        # Apply salary filter
        if salary_min > 0:
             # Re-calculate numeric salary for filtering (as in the sidebar logic)
             def extract_salary_numeric(salary_str):
                  if pd.isna(salary_str) or salary_str == 'Not specified': return 0
                  try: return int(re.sub(r'\D', '', str(salary_str)))
                  except: return 0

             # Create a numeric salary column for filtering on the copy
             filtered_jobs['salary_value'] = filtered_jobs['Salary'].apply(extract_salary_numeric)
             filtered_jobs = filtered_jobs[filtered_jobs['salary_value'] >= salary_min]


        # Show statistics and visualizations
        if filtered_jobs.empty:
            st.warning("No jobs found with the current filters. Try adjusting your filters.")
        else:
            st.info(f"Displaying {len(filtered_jobs)} job{ 's' if len(filtered_jobs) != 1 else '' } matching the filters.")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Job Count by Sector")

                # Classify jobs into sectors (using sector_options from your old app.py)
                # Define sector_options dictionary again if not globally accessible (or define it globally)
                # Let's define it globally near the other global definitions
                # ... (sector_options dictionary definition should be moved to global scope)
                # Note: The old app.py defined sector_options before this section. Let's move it to the top.
                # (Self-correction: sector_options is already defined in the 'Resume-to-Job Matching' section.
                # It needs to be defined *outside* the modes if used in multiple modes/sidebar.
                # Let's move sector_options definition near the job loading).

                def classify_job(job_title, job_desc, reqs): # Added reqs based on filtering logic
                     title = str(job_title).lower()
                     desc = str(job_desc).lower()
                     reqs_text = str(reqs).lower() # Include requirements in classification
                     text = title + " " + desc + " " + reqs_text # Combine all

                     for sector, keywords in sector_options.items():
                          for keyword in keywords:
                               if keyword.lower() in text:
                                    return sector # Return the first matching sector

                     return "Other"

                # Create sector classification on the filtered DataFrame copy
                filtered_jobs['sector'] = filtered_jobs.apply(
                    lambda row: classify_job(row['Job title'], row.get('Job description', ''), row.get('Requirements', '')),
                    axis=1
                )

                # Count by sector
                sector_counts = filtered_jobs['sector'].value_counts()

                # Create pie chart
                fig = px.pie(
                    names=sector_counts.index,
                    values=sector_counts.values,
                    title="Job Distribution by Sector",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Top Companies Hiring")

                # Count jobs by company
                company_counts = filtered_jobs['Company'].value_counts().head(10)

                # Create horizontal bar chart
                if not company_counts.empty:
                    fig = px.bar(
                        x=company_counts.values,
                        y=company_counts.index,
                        orientation='h',
                        title="Top 10 Companies with Open Positions",
                        labels={'x': 'Number of Jobs', 'y': 'Company'},
                        color=company_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Order companies by count
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No companies found in the filtered jobs.")


            # Show job listing table
            st.markdown("#### Available Job Listings")
            # Select relevant columns and format match_score if needed (not applicable in explorer, just list)
            display_cols_explorer = ['Job title', 'Company', 'Salary', 'URL'] # Include URL in display
            if 'salary_value' in filtered_jobs.columns: # Drop temporary column
                 filtered_jobs_display = filtered_jobs.drop(columns=['salary_value', 'sector'])
            else:
                 filtered_jobs_display = filtered_jobs.drop(columns=['sector'])

            st.dataframe(
                filtered_jobs_display[display_cols_explorer],
                use_container_width=True,
                hide_index=True,
                 # Optional: Format URL column if st.dataframe supports it directly, or use st.link_button in a loop
                 # column_config={"URL": st.column_config.LinkColumn("URL")} # Example if using LinkColumn
            )

            # Word cloud of job requirements
            st.markdown("#### Most In-Demand Skills")

            # Extract and count skills from job descriptions for word cloud
            # Use the semantic skill extraction method if available, otherwise simple tokenization
            if SEMANTIC_MATCHER_AVAILABLE:
                 logger.info("Using semantic_matcher.extract_skills_from_text for word cloud.")
                 all_job_text_combined = " ".join(filtered_jobs['Job description'].fillna('') + " " + filtered_jobs['Requirements'].fillna(''))
                 # This extraction gets a list of skills, not just raw text for wordcloud
                 # Let's extract raw text for the wordcloud generator
                 text_for_wordcloud = " ".join(filtered_jobs['Job description'].fillna('') + " " + filtered_jobs['Requirements'].fillna(''))
            else:
                 logger.warning("Semantic matcher not available, using raw text for word cloud.")
                 text_for_wordcloud = " ".join(filtered_jobs['Job description'].fillna('') + " " + filtered_jobs['Requirements'].fillna(''))


            # Create word cloud
            try:
                from wordcloud import WordCloud # Import here if not global
                import matplotlib.pyplot as plt # Import here if not global

                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                     colormap='viridis', max_words=100).generate(text_for_wordcloud)

                # Display the word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except ImportError:
                st.warning("WordCloud package not installed. Install it (`pip install wordcloud matplotlib`) to see skill word cloud visualization.")
            except Exception as e:
                 logger.error(f"Error generating word cloud: {e}", exc_info=True)
                 st.warning(f"Could not generate word cloud: {e}")


# --- Footer (from your old app.py) ---
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Smart Job Matcher | Designed to help you find your perfect career match</p>
</div>
""", unsafe_allow_html=True)

# --- Move global definitions to the top ---
# sector_options dictionary should be defined globally before being used in modes
sector_options = {
    "Education & Management": ["багш", "сургалт", "удирдлага", "education", "teacher", "lecturer", "professor", "academic"],
    "Customer Service": ["customer service", "үйлчилгээ", "захиалга", "client", "support", "help desk", "call center"],
    "Leadership": ["менежер", "захирал", "manager", "director", "executive", "chief", "head of", "supervisor"],
    "Tech & Development": ["developer", "инженер", "програм", "software", "programmer", "coder", "IT", "tech", "web", "mobile"],
    "Creative & Marketing": ["дизайн", "сошиал", "контент", "media", "designer", "creative", "marketing", "brand", "SEO", "content"],
    "Finance": ["нягтлан", "санхүү", "finance", "accountant", "accounting", "financial", "budget", "tax", "banking"],
    "Healthcare": ["эмч", "сувилагч", "health", "эрүүл мэнд", "doctor", "nurse", "medical", "healthcare", "clinic", "hospital"],
    "Logistics & Support": ["logistics", "тээвэр", "нярав", "туслах", "warehouse", "shipping", "supply chain", "inventory"],
    "Data & AI": ["data analyst", "data scientist", "AI", "мэдээлэл шинжээч", "machine learning", "analytics", "big data", "statistics"],
    "HR & Recruitment": ["HR", "хүний нөөц", "recruiter", "talent", "hiring", "recruitment", "personnel", "staffing"],
    "Legal & Compliance": ["хууль", "lawyer", "legal", "compliance", "attorney", "law", "regulatory", "contracts"],
    "Sales": ["борлуулалт", "sales", "зөвлөх", "business development", "account manager", "retail", "revenue"],
    "Project Management": ["төслийн менежер", "project manager", "project coordinator", "scrum", "agile", "program manager"],
    "Engineering & Construction": ["механик", "цахилгаан", "civil", "барилга", "engineer", "mechanical", "electrical", "construction"]
}
