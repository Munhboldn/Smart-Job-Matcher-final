import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import time
import logging
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Define stopwords once
stopwords = set(STOPWORDS)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Smart Job Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Smart Job Matcher\nFind the perfect job match for your skills and experience!"
    }
)

# --- Logging ---
logger = logging.getLogger("smart_job_matcher")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# --- Import Modules ---
try:
    from resume_parser import extract_text_from_pdf, extract_text_from_docx, analyze_resume, summarize_resume
except ImportError:
    st.error("Resume parser module is missing.")
    st.stop()

try:
    from semantic_matcher import semantic_match_resume, get_skill_matches
    SEMANTIC_OK = True
except ImportError:
    SEMANTIC_OK = False
    st.warning("Semantic matcher not available.")

# --- Session State Init ---
def init_session():
    for key, default in {
        'resume_text': '',
        'resume_analysis': None,
        'skills_extracted': [],
        'job_results': None,
        'summary': None,
        'selected_sectors': [],
        'salary_min': 0
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
init_session()

# --- Sidebar Filters ---
if st.session_state.resume_text:
    st.sidebar.text_area("📄 Resume Preview", st.session_state.resume_text[:1000], height=120)

with st.sidebar.expander("📂 Upload Resume", expanded=True):
    resume_file = st.file_uploader("Choose a resume file", type=["pdf", "docx"])
    if resume_file:
        if resume_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_docx(resume_file)
        st.session_state.resume_text = resume_text
        st.session_state.resume_analysis = analyze_resume(resume_text)
        st.session_state.summary = summarize_resume(resume_text)

with st.sidebar.expander("🎛️ Job Filters", expanded=False):
    st.session_state['selected_sectors'] = st.multiselect(
        "Select sectors you're interested in:",
        ["Education", "Tech", "Healthcare", "Finance", "HR", "Sales", "Marketing", "Engineering"]
    )
    st.session_state['salary_min'] = st.number_input("Minimum Salary (₮)", 0, 10000000, 0, step=100000)

# --- App Mode ---
app_mode = st.sidebar.radio("Choose View", ["Resume Analysis", "Job Matching", "Job Market Explorer"])

# --- Resume Analysis ---
if app_mode == "Resume Analysis" and st.session_state.resume_text:
    st.title("📊 Resume Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Content", "Tips", "Summary"])

    with tab1:
        score = st.session_state.resume_analysis.get("completeness_score", 0)
        st.metric("Completeness Score", f"{score}%")
        st.progress(score / 100)

    with tab2:
        st.subheader("Detected Sections")
        for sec in st.session_state.resume_analysis.get("sections", {}).keys():
            st.markdown(f"- {sec.title()}")

        st.subheader("Detected Skills")
        st.write(", ".join(st.session_state.resume_analysis.get("skills", [])[:20]))

    with tab3:
        st.markdown("""
        - Use action verbs (e.g., Led, Developed, Managed)
        - Quantify achievements (e.g., Increased revenue by 15%)
        - Ensure section titles are clear and consistent
        - Keep formatting clean and easy to scan
        """)

    with tab4:
        summary = st.session_state.summary
        st.metric("Estimated Years of Experience", summary["years_of_experience"])
        st.subheader("Roles Mentioned")
        st.write(", ".join(summary.get("roles", [])))
        st.subheader("Top Skills")
        st.write(", ".join(summary.get("top_skills", [])))

# --- Job Matching ---
if app_mode == "Job Matching" and st.session_state.resume_text:
    st.title("🔍 Resume-to-Job Matching")
    jobs_df = pd.read_csv("data/zangia_filtered_jobs.csv")

    # Filter jobs
    if st.session_state.selected_sectors:
        sector_keywords = {
            "Education": ["education", "teacher", "багш"],
            "Tech": ["developer", "software", "инженер"],
            "Healthcare": ["nurse", "doctor", "эмч"],
            "Finance": ["accountant", "санхүү", "нягтлан"],
            "HR": ["HR", "recruitment", "хүний нөөц"],
            "Sales": ["sales", "борлуулалт"],
            "Marketing": ["marketing", "контент", "brand"],
            "Engineering": ["engineer", "construction", "барилга"]
        }
        keywords = [kw for s in st.session_state.selected_sectors for kw in sector_keywords.get(s, [])]
        pattern = '|'.join(keywords)
        jobs_df = jobs_df[
            jobs_df['Job title'].str.lower().str.contains(pattern, na=False) |
            jobs_df['Job description'].str.lower().str.contains(pattern, na=False)
        ]

    if st.session_state.salary_min > 0:
        def extract_salary_value(s):
            import re
            nums = re.findall(r'\d+', str(s))
            return int(nums[0]) if nums else 0
        jobs_df['numeric_salary'] = jobs_df['Salary'].apply(extract_salary_value)
        jobs_df = jobs_df[jobs_df['numeric_salary'] >= st.session_state.salary_min]

    if SEMANTIC_OK:
        matches_df = semantic_match_resume(st.session_state.resume_text, jobs_df, top_n=10)
        st.session_state.job_results = matches_df

        st.subheader("✅ Top Matching Jobs")
        if st.session_state.summary:
            st.success("🏅 Your resume looks great! Consider applying widely.")
        for _, row in matches_df.iterrows():
            st.markdown(f"### {row['Job title']} at {row['Company']}")
            from nltk.corpus import wordnet

def expand_keywords(keywords):
    expanded = set(keywords)
    for word in keywords:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if '_' not in lemma.name():
                    expanded.add(lemma.name().lower())
    return list(expanded)

expanded_skills = expand_keywords(st.session_state.summary['top_skills'])
matched = get_skill_matches(expanded_skills, row['Job description'])
            description = row['Job description']
            for skill in matched:
                description = description.replace(skill, f"<mark>{skill}</mark>")
            st.markdown(description, unsafe_allow_html=True)
            st.caption(f"Score: {row['match_score']:.2f} | [Apply Here]({row['URL']})")

        st.download_button(
            label="📥 Download Results as CSV",
            data=matches_df.to_csv(index=False).encode("utf-8"),
            file_name="matched_jobs.csv",
            mime="text/csv"
        )

# --- Job Market Explorer ---
if app_mode == "Job Market Explorer":
    st.header("📊 Job Market Explorer")
    df = pd.read_csv("data/zangia_filtered_jobs.csv")
    st.subheader("Job Distribution by Sector")
    df['sector'] = df['Job title'].fillna('') + ' ' + df['Job description'].fillna('')
    df['sector'] = df['sector'].str.lower().apply(lambda x: 'tech' if 'developer' in x else 'other')
    fig = px.pie(df['sector'].value_counts().reset_index(), names='index', values='sector', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Hiring Companies")
    top_companies = df['Company'].value_counts().head(10)
    st.bar_chart(top_companies)

    st.subheader("Word Cloud of Job Descriptions")
    text = ' '.join(df['Job description'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
