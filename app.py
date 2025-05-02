import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
import time
from resume_parser import extract_text_from_pdf, extract_text_from_docx, analyze_resume, extract_resume_sections
from semantic_matcher import (
    semantic_match_resume, 
    extract_resume_keywords, 
    extract_skills_from_resume,
    get_skill_matches,
    highlight_matched_skills,
    analyze_resume as analyze_resume_skills
)

# Page configuration
st.set_page_config(
    page_title="Smart Job Matcher", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Smart Job Matcher\nFind the perfect job match for your skills and experience!"
    }
)

# Inject custom styling
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
    .match-score-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .match-score-medium {
        color: #FF8F00;
        font-weight: bold;
    }
    .match-score-low {
        color: #C62828;
        font-weight: bold;
    }
    mark {
        background-color: #FFEB3B;
        padding: 0.1em;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'resume_analysis' not in st.session_state:
    st.session_state.resume_analysis = None
if 'job_results' not in st.session_state:
    st.session_state.job_results = None
if 'skills_extracted' not in st.session_state:
    st.session_state.skills_extracted = []

# Header
st.markdown('<div class="main-header">💼 Smart Job Matcher</div>', unsafe_allow_html=True)

# App Mode
app_mode = st.sidebar.radio("Select Mode", ["Resume-to-Job Matching", "Resume Analysis", "Job Market Explorer"])

# Load jobs
@st.cache_data(ttl=3600)
def load_jobs():
    df = pd.read_csv("data/zangia_filtered_jobs.csv")
    df['Salary'] = df['Salary'].fillna('Not specified')
    df['Job description'] = df['Job description'].fillna('')
    df['Requirements'] = df['Requirements'].fillna('')
    return df

jobs_df = load_jobs()

# Resume-to-Job Matching View
if app_mode == "Resume-to-Job Matching":
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            st.session_state.resume_text = resume_text
            resume_analysis = analyze_resume(resume_text)
            st.session_state.resume_analysis = resume_analysis
            st.session_state.skills_extracted = resume_analysis.get("skills", [])

            with st.expander("View Extracted Resume Text"):
                st.text_area("Resume Text", resume_text, height=200)

    with col2:
        if st.session_state.resume_analysis:
            st.markdown("### 📊 Resume Stats")
            completeness = resume_analysis.get("completeness", 0)
            st.metric("Completeness", f"{completeness:.0f}%")
            st.progress(completeness / 100)

            st.markdown("**Sections Detected:**")
            for section in resume_analysis.get("sections", {}):
                st.markdown(f"- {section.title()}")

            st.markdown("**Extracted Skills:**")
            st.markdown(", ".join(st.session_state.skills_extracted[:10]))

    if st.session_state.resume_text:
        st.markdown("### 🔍 Job Matching")

        # Grouped Sector Filters
        sector_options = {
            "Education & Management": ["багш", "сургалт", "удирдлага", "education", "teacher", "academic"],
            "Customer Service": ["customer service", "үйлчилгээ", "client", "support"],
            "Leadership": ["менежер", "захирал", "manager", "director"],
            "Tech & Development": ["developer", "инженер", "software", "IT", "tech"],
            "Finance": ["нягтлан", "санхүү", "finance", "accountant", "tax"],
            "Creative & Marketing": ["дизайн", "сошиал", "контент", "marketing", "media"],
            "Healthcare": ["эмч", "сувилагч", "medical", "clinic"],
            "Logistics": ["logistics", "тээвэр", "warehouse"],
            "HR & Recruitment": ["HR", "recruiter", "talent"],
            "Sales": ["борлуулалт", "sales", "retail"]
        }

        selected_sectors = st.multiselect("Filter by Sector(s)", list(sector_options.keys()))
        selected_keywords = [kw for sector in selected_sectors for kw in sector_options[sector]]

        if selected_keywords:
            pattern = '|'.join(selected_keywords)
            jobs_df = jobs_df[
                jobs_df['Job title'].str.lower().str.contains(pattern, na=False) |
                jobs_df['Job description'].str.lower().str.contains(pattern, na=False)
            ]

        top_n = st.slider("Number of job matches", 5, 30, 10)

        if st.button("🔍 Match Jobs"):
            with st.spinner("Matching your resume to jobs..."):
                results = semantic_match_resume(
                    st.session_state.resume_text,
                    jobs_df,
                    top_n=top_n,
                    highlight=True,
                    save_csv=True
                )
                st.session_state.job_results = results

            st.success(f"Found {len(results)} matching jobs.")
            st.plotly_chart(px.bar(
                results,
                x='Job title',
                y='match_score',
                color='match_score',
                color_continuous_scale='viridis'
            ), use_container_width=True)

            tab1, tab2, tab3 = st.tabs(["List View", "Detailed View", "Resume Tips"])
            with tab1:
                df_display = results[['Job title', 'Company', 'Salary', 'match_score']].copy()
                df_display['match_score'] = df_display['match_score'].round(1).astype(str) + '%'
                st.dataframe(df_display, use_container_width=True)

            with tab2:
                for i, (_, row) in enumerate(results.iterrows()):
                    job_text = str(row.get("Job description", "")) + " " + str(row.get("Requirements", ""))
                    matched_skills, missing_skills = get_skill_matches(st.session_state.skills_extracted, job_text)
                    highlighted_desc = highlight_matched_skills(job_text, matched_skills)

                    st.markdown(f"### {i+1}. {row['Job title']}")
                    st.markdown(f"**Company:** {row.get('Company', 'Unknown')}")
                    st.markdown(f"**Salary:** {row.get('Salary', 'Not specified')}")
                    st.markdown(f"**Match Score:** <span class='match-score-high'>{row['match_score']:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"[🔗 View Job Posting]({row['URL']})")

                    if matched_skills:
                        st.markdown(f"🟢 **Matched skills:** {', '.join(matched_skills)}")
                    if missing_skills:
                        st.markdown(f"🔴 **Missing skills:** {', '.join(missing_skills)}")

                    with st.expander("View Job Details"):
                        st.markdown(highlighted_desc, unsafe_allow_html=True)

            st.download_button()

with tab3:
    st.markdown("### 💡 Tips to Improve Your Resume")

    # Missing Skills Suggestions
    all_missing_skills = []
    for _, row in results.iterrows():
        job_text = str(row.get("Job description", "")) + " " + str(row.get("Requirements", ""))
        _, missing = get_skill_matches(st.session_state.skills_extracted, job_text)
        all_missing_skills.extend(missing)

    from collections import Counter
    common_missing = Counter(all_missing_skills).most_common(5)

    if common_missing:
        st.markdown("**Top Skills to Add:**")
        for skill, count in common_missing:
            st.markdown(f"- **{skill}** (mentioned in {count} job{'s' if count > 1 else ''})")

    st.markdown("### 📝 General Resume Improvement Suggestions")
    st.markdown("""
    - Use strong action verbs to start bullet points in your experience section (e.g., **Managed**, **Developed**, **Led**, **Implemented**).
    - Quantify your achievements with numbers and metrics whenever possible (e.g., "Increased sales by 15%", "Managed a team of 5", "Reduced costs by $10k").
    - Tailor your resume, especially the summary/objective and skills sections, to align with the specific keywords and requirements of the jobs you are targeting.
    - Proofread your resume meticulously for typos and grammatical errors. Consider using a tool like Grammarly.
    - Ensure consistent formatting, fonts (use standard, professional fonts like Arial, Calibri, Times New Roman, Georgia, Verdana), and spacing throughout the document for readability.
    - Keep your resume concise and focused. Typically 1 page for less than 10 years of experience, and a maximum of 2 pages for more extensive experience.
    - Consider adding a professional summary or objective statement at the top, tailored to the type of job you seek.
    """)

    st.markdown("### 🤖 Applicant Tracking System (ATS) Optimization Tips")
    st.markdown("""
    - **Use Standard Section Headings**: Use clear, common titles like "Education", "Experience", "Skills", "Projects", etc.
    - **Include Keywords**: Mirror the language and keywords used in the job description throughout your resume, especially in skills and experience sections.
    - **Simple Formatting**: Avoid complex layouts, tables, text boxes, headers/footers (some systems struggle with these), and heavy graphics. Use bullet points effectively.
    - **Standard Fonts**: Stick to widely recognized fonts (Arial, Calibri, Times New Roman, Georgia, Verdana) and a font size between 10-12pt.
    - **Reverse Chronological Order**: List your experience and education with the most recent items first.
    - **File Type**: Save your resume as a PDF or DOCX. PDF is often preferred for maintaining formatting, but check the application instructions.
    """)")
st.markdown("""
- **Use Standard Section Headings**: Use clear, common titles like "Education", "Experience", "Skills", "Projects", etc.
- **Include Keywords**: Mirror the language and keywords used in the job description throughout your resume, especially in skills and experience sections.
- **Simple Formatting**: Avoid complex layouts, tables, text boxes, headers/footers (some systems struggle with these), and heavy graphics. Use bullet points effectively.
- **Standard Fonts**: Stick to widely recognized fonts (Arial, Calibri, Times New Roman, Georgia, Verdana) and a font size between 10-12pt.
- **Reverse Chronological Order**: List your experience and education with the most recent items first.
- **File Type**: Save your resume as a PDF or DOCX. PDF is often preferred for maintaining formatting, but check the application instructions.
""")
        """)
                mime="text/csv"
            )

# Resume Analysis View
elif app_mode == "Resume Analysis":
    st.subheader("📊 Resume Analyzer")
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        with st.spinner("Analyzing your resume..."):
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            resume_analysis = analyze_resume(resume_text)
            skills_analysis = analyze_resume_skills(resume_text)

        st.subheader("Sections Detected")
        for section, content in resume_analysis.get("sections", {}).items():
            if content.strip():
                with st.expander(section.title()):
                    st.markdown(content)

        st.subheader("Top Keywords")
        keywords = skills_analysis.get("keywords", [])[:20]
        keyword_freq = {kw: keywords.count(kw) for kw in set(keywords)}
        sorted_kws = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)

        fig = go.Figure(go.Bar(
            x=[v for _, v in sorted_kws],
            y=[k for k, _ in sorted_kws],
            orientation='h'
        ))
        fig.update_layout(title="Keyword Frequency", xaxis_title="Count", yaxis_title="Keyword")
        st.plotly_chart(fig, use_container_width=True)

# Job Market Explorer View
elif app_mode == "Job Market Explorer":
    st.subheader("📈 Job Market Explorer")
    salary_min = st.sidebar.number_input("Minimum Salary (₮)", 0, 10000000, 0, 100000)

    def extract_salary(s):
        import re
        nums = re.findall(r'\d+', str(s))
        return int(sum(map(int, nums))/len(nums)) if nums else 0

    jobs_df['numeric_salary'] = jobs_df['Salary'].apply(extract_salary)
    filtered_jobs = jobs_df[jobs_df['numeric_salary'] >= salary_min]

    fig1 = px.pie(
        names=filtered_jobs['Job title'].str.extract('(\w+)')[0].value_counts().index,
        values=filtered_jobs['Job title'].str.extract('(\w+)')[0].value_counts().values,
        title="Job Distribution by Sector"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        x=filtered_jobs['Company'].value_counts().head(10).values,
        y=filtered_jobs['Company'].value_counts().head(10).index,
        orientation='h',
        title="Top Companies Hiring"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(filtered_jobs[['Job title', 'Company', 'Salary']], use_container_width=True)
