# Smart Job Matcher App (Enhanced UI with Full Features)
# Includes matching, analysis, word cloud, tabs, and Excel download

# --- Imports and Config ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time
import logging
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# --- Safe Imports ---
try:
    from resume_parser import extract_text_from_pdf, extract_text_from_docx, analyze_resume
except ImportError:
    st.error("Resume parser module is missing.")
    st.stop()

try:
    from semantic_matcher import (
        semantic_match_resume,
        analyze_resume as analyze_resume_skills,
        get_skill_matches
    )
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
        'job_results': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
init_session()

# --- Load Jobs ---
@st.cache_data(ttl=3600)
def load_jobs():
    try:
        df = pd.read_csv("data/zangia_filtered_jobs.csv")
        df['Salary'] = df['Salary'].fillna('Not specified')
        df['Job description'] = df['Job description'].fillna('')
        df['Requirements'] = df['Requirements'].fillna('')
        return df
    except Exception as e:
        st.error(f"Failed to load job data: {e}")
        return pd.DataFrame(columns=['Job title', 'Company', 'Salary', 'Job description', 'Requirements', 'URL'])

jobs_df = load_jobs()

# --- Define sector options (used in both modes) ---
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

# --- Sidebar ---
st.sidebar.header("⚙️ Options")
app_mode = st.sidebar.radio("Select Mode", ["Resume-to-Job Matching", "Resume Analysis", "Job Market Explorer"])
salary_min = st.sidebar.number_input("Minimum Salary (₮)", 0, 10000000, 0, step=100000)
st.sidebar.markdown("---")
st.sidebar.write("#### About")
st.sidebar.write("Smart Job Matcher helps you find jobs that match your skills and experience. Upload your resume and get personalized job recommendations.")

# --- Header ---
st.markdown('<div class="main-header">💼 Smart Job Matcher</div>', unsafe_allow_html=True)

# --- Resume-to-Job Matching ---
if app_mode == "Resume-to-Job Matching":
    st.header("📄 Upload Your Resume")
    resume_file = st.file_uploader("", type=["pdf", "docx"])

    if resume_file:
        with st.spinner("Processing resume..."):
            if resume_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(resume_file)
            else:
                text = extract_text_from_docx(resume_file)
            st.session_state.resume_text = text
            st.session_state.resume_analysis = analyze_resume(text)
            st.session_state.skills_extracted = analyze_resume_skills(text).get("skills", [])

        st.success("Resume uploaded and processed successfully.")

        # Sector filter added here
        st.subheader("🎯 Filter Job Results by Sector(s)")
        selected_sectors = st.multiselect("Choose one or more sectors:", list(sector_options.keys()))
        selected_keywords = [kw for sector in selected_sectors for kw in sector_options[sector]] if selected_sectors else []

        if st.button("🔍 Find Matching Jobs"):
        from datetime import datetime
        output = io.BytesIO()
        if not filtered_jobs.empty:
            from fpdf import FPDF
            df_feedback = filtered_jobs[['Job title', 'Company', 'Salary', 'Job description', 'Requirements', 'URL']].copy()
            df_feedback['Matched Skills'] = ', '.join(st.session_state.skills_extracted[:10])

            # Excel Export
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_feedback.to_excel(writer, index=False, sheet_name="Job Matches")
            output.seek(0)

            st.download_button(
                label="⬇ Download Matched Jobs as Excel",
                data=output,
                file_name=f"job_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # PDF Export
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Smart Job Matcher - Matched Jobs Report", ln=True, align='C')
            for index, row in df_feedback.iterrows():
                pdf.multi_cell(0, 10, txt=f"{row['Job title']} at {row['Company']}
Salary: {row['Salary']}
URL: {row['URL']}
", border=0)
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)

            st.download_button(
                label="⬇ Download Matched Jobs as PDF",
                data=pdf_output,
                file_name=f"job_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
                label="⬇ Download Matched Jobs as Excel",
                data=output,
                file_name=f"job_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            filtered_jobs = jobs_df.copy()
            if selected_keywords:
                pattern = '|'.join(selected_keywords)
                filtered_jobs = filtered_jobs[
                    filtered_jobs['Job title'].str.lower().str.contains(pattern, na=False) |
                    filtered_jobs['Job description'].str.lower().str.contains(pattern, na=False)
                ]

            if SEMANTIC_OK:
                results = semantic_match_resume(text, filtered_jobs, top_n=10)
                st.session_state.job_results = results

                st.subheader("✅ Top Matching Jobs")
                st.dataframe(results[['Job title', 'Company', 'Salary', 'match_score', 'URL']])

                # Optional: Display top missing sector skills
                if selected_sectors:
                    st.markdown("### 🔧 Sector-Specific Resume Match")
                    sector_keywords = selected_keywords
                    matched_lower = [s.lower() for s in matched]
                    total_sector = len(sector_keywords)
                    matched_sector = len([kw for kw in sector_keywords if kw.lower() in matched_lower])
                    match_percent = int((matched_sector / total_sector) * 100) if total_sector > 0 else 0
                    st.metric("Sector Keyword Coverage", f"{match_percent}%")
                    st.progress(match_percent / 100)

                    missing_sector = [kw for kw in sector_keywords if kw.lower() not in matched_lower]
                    st.markdown("### 🔧 Sector-Specific Resume Match")
                    sector_keywords = selected_keywords
                    missing_sector = [kw for kw in sector_keywords if kw.lower() not in [s.lower() for s in matched]]
                    if missing_sector:
                        st.warning(f"You might be missing {len(missing_sector)} keywords important for these sectors:")
                        for ms in missing_sector[:10]:
                            st.markdown(f"- {ms}")
                    else:
                        st.success("Great! Your resume covers the main keywords for the selected sectors.")

# --- Resume Analysis ---
from datetime import datetime
st.sidebar.subheader("🎯 Target Sector")
selected_sector_analysis = st.sidebar.selectbox("Choose sector for feedback:", [None] + list(sector_options.keys()))

if app_mode == "Resume Analysis":
    uploaded_file = st.file_uploader("Upload your resume for analysis", type=["pdf", "docx"])
    if uploaded_file:
        with st.spinner("Analyzing resume..."):
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            st.session_state.resume_text = resume_text
            st.session_state.resume_analysis = analyze_resume(resume_text)
            skills_data = analyze_resume_skills(resume_text)
            st.session_state.skills_extracted = skills_data.get("skills", [])
            keywords = skills_data.get("keywords", [])

        st.markdown("### 📊 Resume Insights")
        tab1, tab2, tab3 = st.tabs(["Overview", "Content Analysis", "Improvement Tips"])

        with tab1:
            completeness = st.session_state.resume_analysis.get("completeness_score", 0)
            st.metric("Completeness Score", f"{completeness}%")
            st.progress(completeness / 100)
            st.write("**Detected Sections:**")
            for section in st.session_state.resume_analysis.get("sections", {}):
                st.markdown(f"- {section.title()}")

        with tab2:
            st.write("**Skills Detected:**")
            if st.session_state.skills_extracted:
                st.markdown(", ".join(st.session_state.skills_extracted[:20]))
            else:
                st.warning("No specific skills detected.")

            if keywords:
                keyword_freq = Counter(keywords).most_common(15)
                fig = go.Figure(go.Bar(
                    x=[v for _, v in keyword_freq],
                    y=[k for k, _ in keyword_freq],
                    orientation='h',
                    marker_color='royalblue'
                ))
                fig.update_layout(
                    title="Top Keywords",
                    xaxis_title="Frequency",
                    yaxis_title="Keyword",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.write("**Resume Tips:**")

            if selected_sector_analysis:
                st.subheader("📄 Download Resume Feedback Report")
                feedback_lines = [
                    f"Sector: {selected_sector_analysis}",
                    f"Matched Keywords: {', '.join(matched_sector_keywords) if matched_sector_keywords else 'None'}",
                    f"Missing Keywords: {', '.join(missing_sector_keywords) if missing_sector_keywords else 'None'}",
                    f"Completeness Score: {completeness}%",
                    f"Skills Detected: {', '.join(st.session_state.skills_extracted[:10])}"
                ]
                feedback_text = "
".join(feedback_lines)

                # Create downloadable report
                output = io.BytesIO()
                df_feedback = pd.DataFrame({"Feedback": feedback_lines})
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_feedback.to_excel(writer, index=False, sheet_name="Resume Feedback")
                output.seek(0)

                st.download_button(
                    label="⬇ Download Feedback as Excel",
                    data=output,
                    file_name=f"resume_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # PDF Export
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Smart Job Matcher - Resume Feedback", ln=True, align='C')
                for line in feedback_lines:
                    pdf.multi_cell(0, 10, txt=line, border=0)
                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                pdf_output.seek(0)

                st.download_button(
                    label="⬇ Download Feedback as PDF",
                    data=pdf_output,
                    file_name=f"resume_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                ).strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # --- General tips ---
            - Use action verbs (e.g., Led, Developed, Managed)
            - Quantify achievements (e.g., "Increased sales by 15%")
            - Keep formatting consistent
            - Tailor your resume for the role
            """)

# --- Job Market Explorer ---
if app_mode == "Job Market Explorer":
    st.header("📊 Job Market Explorer")
    st.subheader("Overview of current job trends")

    # Filter by salary
    filtered_jobs = jobs_df.copy()
    if salary_min > 0:
        def extract_salary(salary_str):
            if pd.isna(salary_str) or salary_str == 'Not specified':
                return 0
            import re
            numbers = re.findall(r'\d+', str(salary_str))
            return sum(map(int, numbers)) / len(numbers) if numbers else 0

        filtered_jobs['salary_value'] = filtered_jobs['Salary'].apply(extract_salary)
        filtered_jobs = filtered_jobs[filtered_jobs['salary_value'] >= salary_min]

    if filtered_jobs.empty:
        st.warning("No jobs found with the current filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Job Count by Sector")
            def classify_job(job_title, job_desc):
                text = (str(job_title) + " " + str(job_desc)).lower()
                for sector, keywords in sector_options.items():
                    for keyword in keywords:
                        if keyword.lower() in text:
                            return sector
                return "Other"

            filtered_jobs['sector'] = filtered_jobs.apply(
                lambda row: classify_job(row['Job title'], row.get('Job description', '')),
                axis=1
            )
            sector_counts = filtered_jobs['sector'].value_counts()
            fig = px.pie(
                names=sector_counts.index,
                values=sector_counts.values,
                title="Job Distribution by Sector",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Top Hiring Companies")
            company_counts = filtered_jobs['Company'].value_counts().head(10)
            fig = px.bar(
                x=company_counts.values,
                y=company_counts.index,
                orientation='h',
                title="Top 10 Hiring Companies",
                labels={'x': 'Jobs', 'y': 'Company'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Available Jobs")
        st.dataframe(filtered_jobs[['Job title', 'Company', 'Salary']], use_container_width=True)

        st.markdown("#### Most In-Demand Skills (Word Cloud)")
        all_text = " ".join(filtered_jobs['Job description'].fillna('') + " " + filtered_jobs['Requirements'].fillna(''))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
