import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import logging
import tempfile
import os

# --- Streamlit Page Config  ---
st.set_page_config(
    page_title="Smart Job Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "# Smart Job Matcher\nFind the perfect job match for your skills and experience!"}
)

# --- Import Modules (AFTER set_page_config) ---
from resume_parser import extract_text_from_pdf, extract_text_from_docx, analyze_resume, summarize_resume

try:
    from semantic_matcher_v2 import semantic_match_resume
    SEMANTIC_OK = True
except ImportError:
    SEMANTIC_OK = False
    st.warning("Semantic matcher not available.")

# --- Logging ---
logger = logging.getLogger("smart_job_matcher")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# --- Session State Init ---
def init_session():
    defaults = {
        'resume_text': '',
        'resume_analysis': None,
        'summary': None,
        'selected_sectors': [],
        'salary_min': 0
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session()

# --- Sidebar ---
with st.sidebar.expander("📂 Upload Resume", expanded=True):
    resume_file = st.file_uploader("Choose a resume file", type=["pdf", "docx"])
    if resume_file:
        if resume_file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(resume_file.getvalue())
                tmp_path = tmp_file.name
            st.session_state.resume_text = extract_text_from_pdf(tmp_path)
            os.unlink(tmp_path)
        else:
            st.session_state.resume_text = extract_text_from_docx(resume_file)

        st.session_state.resume_analysis = analyze_resume(st.session_state.resume_text)
        st.session_state.summary = summarize_resume(st.session_state.resume_text)

with st.sidebar.expander("🎛️ Job Filters", expanded=False):
    st.session_state.selected_sectors = st.multiselect(
        "Select sectors you're interested in:",
        ["Education", "Tech", "Healthcare", "Finance", "HR", "Sales", "Marketing", "Engineering"]
    )
    st.session_state.salary_min = st.number_input("Minimum Salary (₮)", 0, 10000000, 0, step=100000)

app_mode = st.sidebar.radio("Choose View", ["Resume Analysis", "Job Matching", "Job Market Explorer"])

# --- Resume Analysis ---
if app_mode == "Resume Analysis" and st.session_state.resume_text:
    st.title("📊 Resume Analysis")
    analysis = st.session_state.resume_analysis

    tabs = st.tabs(["Overview", "Content", "Tips", "Summary"])
    with tabs[0]:
        score = analysis.get("completeness_score", 0)
        st.metric("Completeness Score", f"{score}%")
        st.progress(score / 100)

    with tabs[1]:
        st.subheader("Detected Sections")
        for sec in analysis["sections"]:
            st.write(f"- {sec.title()}")
        st.subheader("Detected Skills")
        st.write(", ".join(analysis.get("skills", [])[:20]))

    with tabs[2]:
        st.markdown("""
        - Use action verbs
        - Quantify achievements
        - Ensure clear sections
        - Clean formatting
        """)

    with tabs[3]:
        summary = st.session_state.summary
        st.metric("Years of Experience", summary["years_of_experience"])
        st.subheader("Roles Mentioned")
        st.write(", ".join(summary.get("roles", [])))
        st.subheader("Top Skills")
        st.write(", ".join(summary.get("top_skills", [])))

# --- Job Matching ---
if app_mode == "Job Matching" and st.session_state.resume_text:
    st.title("🔍 Resume-to-Job Matching")
    jobs_df = pd.read_csv("data/zangia_filtered_jobs.csv")

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
        jobs_df = jobs_df[jobs_df['Job title'].str.lower().str.contains(pattern, na=False) |
                          jobs_df['Job description'].str.lower().str.contains(pattern, na=False)]

    if st.session_state.salary_min > 0:
        jobs_df = jobs_df[jobs_df['Salary'].str.extract('(\d+)').astype(float).fillna(0) >= st.session_state.salary_min]

    if SEMANTIC_OK:
        matches_df = semantic_match_resume(st.session_state.resume_text, jobs_df)
        for _, row in matches_df.iterrows():
            st.subheader(f"{row['Job title']} at {row['Company']}")
            st.write(row['Job description'])
            st.caption(f"Score: {row['match_score']:.2f} | [Apply Here]({row['URL']})")

# --- Job Market Explorer ---
if app_mode == "Job Market Explorer":
    st.header("📊 Job Market Explorer")
    df = pd.read_csv("data/zangia_filtered_jobs.csv")
    fig = px.pie(df, names=df['Company'].value_counts().index, values=df['Company'].value_counts().values)
    st.plotly_chart(fig, use_container_width=True)

    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['Job description'].dropna()))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
