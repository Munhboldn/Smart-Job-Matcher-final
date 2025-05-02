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
    analyze_resume as analyze_resume_skills
)

# Page configuration with custom theme
st.set_page_config(
    page_title="Smart Job Matcher", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Smart Job Matcher\nFind the perfect job match for your skills and experience!"
    }
)

# Apply custom styling
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

# Initialize session state for storing resume data
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'resume_analysis' not in st.session_state:
    st.session_state.resume_analysis = None
if 'job_results' not in st.session_state:
    st.session_state.job_results = None
if 'skills_extracted' not in st.session_state:
    st.session_state.skills_extracted = []

# Custom header
st.markdown('<div class="main-header">💼 Smart Job Matcher</div>', unsafe_allow_html=True)

# Load job listings with caching
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_jobs():
    try:
        df = pd.read_csv("data/zangia_filtered_jobs.csv")
        # Add additional preprocessing
        df['Salary'] = df['Salary'].fillna('Not specified')
        df['Job description'] = df['Job description'].fillna('')
        df['Requirements'] = df['Requirements'].fillna('')
        return df
    except Exception as e:
        st.error(f"Error loading job data: {str(e)}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['Job title', 'Company', 'Salary', 'Job description', 'Requirements', 'URL'])

# Grouped sector keywords with enhanced categories
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

# Create flat keyword list from the grouped options
all_keywords = sorted(set([kw for group in sector_options.values() for kw in group]))

# Sidebar for filters and settings
with st.sidebar:
    st.header("⚙️ Options")
    
    # App mode selection
    app_mode = st.radio("Select Mode", ["Resume-to-Job Matching", "Resume Analysis", "Job Market Explorer", "Resume Creator"])
 
    
    if app_mode == "Job Market Explorer":
        jobs_df = load_jobs()
        st.subheader("Job Market Filters")
        salary_min = st.number_input("Minimum Salary (\u20ae)", min_value=0, value=0, step=100000)
    else:
        jobs_df = load_jobs()

    st.markdown("---")
    st.write("#### About")
    st.write("""
    Smart Job Matcher helps you find jobs that match your skills and experience using advanced AI matching technology.

    Upload your resume and get personalized job recommendations!
    """)

# === Resume Creator Mode ===
if app_mode == "Resume Creator":
    st.markdown('📄 Resume Creator', unsafe_allow_html=True)
    st.info("Fill in the fields below to generate your resume")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    linkedin = st.text_input("LinkedIn URL")
    summary = st.text_area("Professional Summary", height=100)

    st.markdown("### 🎓 Education")
    education = st.text_area("List your education background", height=150)

    st.markdown("### 💼 Work Experience")
    experience = st.text_area("List your work experience", height=200)

    st.markdown("### 🛠️ Skills")
    skills = st.text_area("List your skills separated by commas")

    if st.button("Generate Resume"):
        from docx import Document
        doc = Document()
        doc.add_heading(name, 0)
        doc.add_paragraph(f"{email} | {phone} | {linkedin}")
        doc.add_heading("Professional Summary", level=1)
        doc.add_paragraph(summary)
        doc.add_heading("Education", level=1)
        doc.add_paragraph(education)
        doc.add_heading("Work Experience", level=1)
        doc.add_paragraph(experience)
        doc.add_heading("Skills", level=1)
        doc.add_paragraph(skills)

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        st.download_button(
            label="📄 Download Resume",
            data=buffer,
            file_name=f"{name}_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

if app_mode == "Resume-to-Job Matching":
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
                
                # Analyze resume
                resume_analysis = analyze_resume(resume_text)
                st.session_state.resume_analysis = resume_analysis
                
                # Extract skills
                skills_analysis = analyze_resume_skills(resume_text)
                st.session_state.skills_extracted = skills_analysis["skills"]
            
            # Display extracted text in an expandable section
            with st.expander("View Extracted Resume Text", expanded=False):
                st.text_area("", resume_text, height=200)
    
    with col2:
        if uploaded_file:
            # Display resume stats
            st.markdown('<div class="sub-header">📊 Resume Stats</div>', unsafe_allow_html=True)
            
            # Resume completeness score
            completeness = st.session_state.resume_analysis.get('completeness_score', 0)
            st.markdown(f"**Resume Completeness:** {completeness}%")
            st.progress(completeness/100)
            
            # Resume sections found
            st.markdown("**Detected Sections:**")
            sections = st.session_state.resume_analysis.get('sections', {})
            for section in sections:
                if section != 'other' and section != 'content':
                    st.markdown(f"- {section.title()}")
            
            # Skills extracted
            st.markdown("**Extracted Skills:**")
            if st.session_state.skills_extracted:
                skills_text = ", ".join(st.session_state.skills_extracted[:10])
                if len(st.session_state.skills_extracted) > 10:
                    skills_text += f" (+{len(st.session_state.skills_extracted) - 10} more)"
                st.markdown(skills_text)
            else:
                st.markdown("No skills detected. Consider adding more specific skills to your resume.")
    
    # Job matching section
    if uploaded_file:
        st.markdown('<div class="sub-header">🔎 Find Matching Jobs</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Job sector filter
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
            top_n = st.slider("Number of results", min_value=5, max_value=30, value=10, step=5)
            
            # Match button
            find_jobs = st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True)
        
        if find_jobs:
            # Show progress and perform matching
            with st.spinner("Matching your profile to jobs..."):
                # Filter by selected keywords if any
                filtered_jobs = jobs_df
                if selected_keywords:
                    pattern = '|'.join(selected_keywords)
                    filtered_jobs = jobs_df[
                        jobs_df["Job title"].str.lower().str.contains(pattern, na=False) |
                        jobs_df["Job description"].str.lower().str.contains(pattern, na=False)
                    ]
                
                # If no jobs match the filters, show a message
                if filtered_jobs.empty:
                    st.warning("No jobs found matching your selected sectors. Try selecting different sectors.")
                else:
                    # Match using semantic embeddings
                    start_time = time.time()
                    results = semantic_match_resume(st.session_state.resume_text, filtered_jobs, top_n=top_n)
                    matching_time = time.time() - start_time
                    
                    # Store results in session state
                    st.session_state.job_results = results
                    
                    # Summary stats
                    st.success(f"Found {len(results)} matching jobs in {matching_time:.2f} seconds")
                    
                    # Create interactive visualizations of match scores
                    fig = px.bar(
                        results,
                        x='Job title',
                        y='match_score',
                        color='match_score',
                        color_continuous_scale='viridis',
                        labels={'match_score': 'Match Score (%)'},
                        title='Job Match Scores'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display job results
        if st.session_state.job_results is not None:
            st.markdown('<div class="sub-header">✅ Top Matching Jobs</div>', unsafe_allow_html=True)
            
            # Get resume skills for matching
            resume_skills = st.session_state.skills_extracted
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["List View", "Detailed View"])
            
            with tab1:
                # Create a dataframe for display
                display_df = st.session_state.job_results[['Job title', 'Company', 'Salary', 'match_score']].copy()
                display_df['match_score'] = display_df['match_score'].round(1).astype(str) + '%'
                st.dataframe(display_df, use_container_width=True)
            
            with tab2:
                # Show detailed job cards
                for i, (_, row) in enumerate(st.session_state.job_results.iterrows()):
                    with st.container():
                        job_text = str(row.get("Job description", "")) + " " + str(row.get("Requirements", ""))
                        
                        # Get matched and missing skills
                        matched_skills, missing_skills = get_skill_matches(resume_skills, job_text)
                        
                        # Format match score with color
                        match_score = row['match_score']
                        if match_score >= 80:
                            match_class = "match-score-high"
                        elif match_score >= 60:
                            match_class = "match-score-medium"
                        else:
                            match_class = "match-score-low"
                        
                        # Job card
                        st.markdown(f'<div class="match-section">', unsafe_allow_html=True)
                        st.markdown(f'<div class="job-title">{i+1}. {row["Job title"]}</div>', unsafe_allow_html=True)
                        st.markdown(f"**Company:** {row.get('Company', 'Unknown')}")
                        st.markdown(f"**Salary:** {row.get('Salary', 'Not specified')}")
                        st.markdown(f'**Match Score:** <span class="{match_class}">{match_score:.1f}%</span>', unsafe_allow_html=True)
                        
                        # Job URL
                        st.markdown(f"[🔗 View Job Posting]({row['URL']})")
                        
                        # Matched and Missing Skills
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
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")
            
            # Download results
            if not st.session_state.job_results.empty:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    st.session_state.job_results.to_excel(writer, index=False)
                output.seek(0)
                
                st.download_button(
                    label="⬇ Download results as Excel",
                    data=output,
                    file_name="matched_jobs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Resume improvement tips
                st.markdown('<div class="sub-header">🚀 Improve Your Resume</div>', unsafe_allow_html=True)
                st.markdown('<div class="tips-section">', unsafe_allow_html=True)
                st.markdown("Based on your job matches, here are tips to improve your resume:")
                
                # Generate dynamic tips based on job results
                all_missing_skills = []
                for _, row in st.session_state.job_results.iterrows():
                    job_text = str(row.get("Job description", "")) + " " + str(row.get("Requirements", ""))
                    _, missing = get_skill_matches(resume_skills, job_text)
                    all_missing_skills.extend(missing)
                
                # Count most common missing skills
                from collections import Counter
                common_missing = Counter(all_missing_skills).most_common(5)
                
                if common_missing:
                    st.markdown("#### Top Skills to Add:")
                    for skill, count in common_missing:
                        st.markdown(f"- **{skill}** (mentioned in {count} job{'s' if count > 1 else ''})")
                
                # General resume improvement tips
                st.markdown("""
                #### General Tips:
                - Make sure your resume highlights your most relevant skills for the targeted positions
                - Use specific keywords from job descriptions in your resume
                - Quantify your achievements with numbers and metrics
                - Use action verbs to describe your experience
                - Tailor your resume for each application
                """)
                st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Resume Analysis":
    st.markdown('<div class="sub-header">📊 Resume Analyzer</div>', unsafe_allow_html=True)
    st.write("Upload your resume to get a detailed analysis of its content and effectiveness.")
    
    # Upload resume for analysis
    uploaded_file = st.file_uploader("", type=["pdf", "docx"])
    
    if uploaded_file:
        # Process the resume
        with st.spinner("Analyzing your resume..."):
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                resume_text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file format.")
                resume_text = ""
            
            # Analyze resume
            resume_analysis = analyze_resume(resume_text)
            skills_analysis = analyze_resume_skills(resume_text)
            
        # Display analysis results
        if resume_text:
            # Create tabs for different analysis views
            tab1, tab2, tab3 = st.tabs(["Overview", "Content Analysis", "Improvement Tips"])
            
            with tab1:
                # Overview metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Completeness score
                    completeness = resume_analysis.get('completeness_score', 0)
                    st.metric("Resume Completeness", f"{completeness}%")
                    st.progress(completeness/100)
                
                with col2:
                    # Section count
                    sections = resume_analysis.get('sections', {})
                    section_count = len([s for s in sections if s not in ['other', 'content']])
                    st.metric("Sections Found", section_count)
                    
                with col3:
                    # Skills count
                    skills_count = len(skills_analysis.get('skills', []))
                    st.metric("Skills Detected", skills_count)
                
                # Contact information
                st.markdown("#### Contact Information")
                contact_info = resume_analysis.get('contact_info', {})
                
                if contact_info.get('emails') or contact_info.get('phones') or contact_info.get('linkedin'):
                    if contact_info.get('emails'):
                        st.write(f"📧 Email: {contact_info['emails'][0]}")
                    if contact_info.get('phones'):
                        st.write(f"📱 Phone: {contact_info['phones'][0]}")
                    if contact_info.get('linkedin'):
                        st.write(f"🔗 LinkedIn: {contact_info['linkedin'][0]}")
                    if contact_info.get('locations'):
                        st.write(f"📍 Location: {', '.join(contact_info['locations'])}")
                else:
                    st.warning("No contact information detected. Make sure your resume includes clear contact details.")
            
            with tab2:
                # Skills analysis
                st.markdown("#### Skills Analysis")
                
                # Display skills as tags
                skills = skills_analysis.get('skills', [])
                if skills:
                    # Create columns for skills display
                    cols = st.columns(3)
                    for i, skill in enumerate(skills):
                        cols[i % 3].markdown(f"- {skill}")
                else:
                    st.warning("No specific skills detected. Consider adding more explicit skills to your resume.")
                
                # Keywords visualization
                st.markdown("#### Keywords")
                keywords = skills_analysis.get('keywords', [])
                if keywords:
                    # Create a word frequency dictionary for visualization
                    keyword_freq = {kw: keywords.count(kw) for kw in set(keywords[:20])}
                    
                    # Sort by frequency
                    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
                    
                    # Create horizontal bar chart
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in sorted_keywords],
                        y=[k for k, _ in sorted_keywords],
                        orientation='h',
                        marker_color='royalblue'
                    ))
                    fig.update_layout(
                        title="Top Keywords in Your Resume",
                        xaxis_title="Frequency",
                        yaxis_title="Keyword",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display resume sections
                st.markdown("#### Resume Sections")
                sections = resume_analysis.get('sections', {})
                
                if sections:
                    for section_name, content in sections.items():
                        if section_name not in ['other', 'content'] and content.strip():
                            with st.expander(f"{section_name.title()} Section"):
                                st.write(content)
                else:
                    st.warning("No clear sections detected in your resume.")
            
            with tab3:
                # Resume improvement suggestions
                st.markdown("#### Resume Improvement Suggestions")
                
                # Check completeness
                if completeness < 70:
                    st.markdown("##### 🔴 Completeness Issues:")
                    missing_sections = []
                    for section in ['contact', 'education', 'experience', 'skills']:
                        if section not in sections or len(sections[section]) < 20:
                            missing_sections.append(section)
                    
                    if missing_sections:
                        st.markdown("Consider adding or expanding these sections:")
                        for section in missing_sections:
                            st.markdown(f"- **{section.title()}** section")
                
                # Skills suggestions
                if len(skills) < 10:
                    st.markdown("##### 🔶 Skills Recommendations:")
                    st.markdown("""
                    - Add more specific skills relevant to your target industry
                    - Include both technical and soft skills
                    - Consider adding skill levels (e.g., "Proficient in Python")
                    """)
                
                # Content analysis
                st.markdown("##### 📝 Content Tips:")
                if len(resume_text) < 1500:
                    st.markdown("- Your resume seems brief. Consider adding more detail to your experiences.")
                
                # General tips
                st.markdown("##### 💡 General Improvements:")
                st.markdown("""
                - Use action verbs to start bullet points (achieved, created, developed, etc.)
                - Quantify your achievements with numbers where possible
                - Ensure consistent formatting throughout the document
                - Proofread for spelling and grammar errors
                - Tailor your resume for specific job applications
                """)
                
                # Resume samples suggestion
                st.markdown("##### 📚 Additional Resources:")
                st.markdown("""
                - Review sample resumes in your industry for inspiration
                - Consider having your resume reviewed by a professional
                - Use industry-specific keywords found in job descriptions
                """)

elif app_mode == "Job Market Explorer":
    st.markdown('<div class="sub-header">🔍 Job Market Explorer</div>', unsafe_allow_html=True)
    st.write("Explore the current job market trends and opportunities.")
    
    # Apply any filters from sidebar
    filtered_jobs = jobs_df
    
    # Apply salary filter if set
    if 'salary_min' in locals() and salary_min > 0:
        # Extract numeric salary values where possible
        def extract_salary(salary_str):
            if pd.isna(salary_str) or salary_str == 'Not specified':
                return 0
            # Extract numbers from the string
            import re
            numbers = re.findall(r'\d+', str(salary_str))
            if numbers:
                # If multiple numbers, take the average
                return sum(map(int, numbers)) / len(numbers)
            return 0
        
        # Create a numeric salary column for filtering
        filtered_jobs['salary_value'] = filtered_jobs['Salary'].apply(extract_salary)
        filtered_jobs = filtered_jobs[filtered_jobs['salary_value'] >= salary_min]
    
    # Show statistics and visualizations
    if filtered_jobs.empty:
        st.warning("No jobs found with the current filters. Try adjusting your filters.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Job Count by Sector")
            
            # Classify jobs into sectors
            def classify_job(job_title, job_desc):
                title = str(job_title).lower()
                desc = str(job_desc).lower()
                text = title + " " + desc
                
                for sector, keywords in sector_options.items():
                    for keyword in keywords:
                        if keyword.lower() in text:
                            return sector
                return "Other"
            
            # Create sector classification
            filtered_jobs['sector'] = filtered_jobs.apply(
                lambda row: classify_job(row['Job title'], row.get('Job description', '')), 
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
            fig = px.bar(
                x=company_counts.values,
                y=company_counts.index,
                orientation='h',
                title="Top 10 Companies with Open Positions",
                labels={'x': 'Number of Jobs', 'y': 'Company'},
                color=company_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show job listing table
        st.markdown("#### Available Job Listings")
        st.dataframe(
            filtered_jobs[['Job title', 'Company', 'Salary']],
            use_container_width=True
        )
        
        # Word cloud of job requirements
        st.markdown("#### Most In-Demand Skills")
        
        # Extract and count skills from job descriptions
        all_job_text = " ".join(filtered_jobs['Job description'].fillna('') + " " + filtered_jobs['Requirements'].fillna(''))
        
        # Create word cloud
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                colormap='viridis', max_words=100).generate(all_job_text)
            
            # Display the word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except ImportError:
            st.warning("WordCloud package not installed. Install it to see skill word cloud visualization.")
            
elif app_mode == "Resume Enhancement":
    st.markdown('<div class="sub-header">✨ Resume Enhancement</div>', unsafe_allow_html=True)
    
    st.markdown("### 📝 General Resume Improvement Suggestions")
    st.markdown("""
- Use **strong action verbs** to start bullet points (e.g., *Managed*, *Developed*, *Led*, *Implemented*).
- **Quantify** achievements with numbers and metrics where possible (e.g., *Increased sales by 15%*).
- Tailor your resume using **keywords from the job description**.
- **Proofread** your resume to remove typos (tools like Grammarly can help).
- Keep formatting **clean and consistent** (fonts, spacing, layout).
- Keep it **concise**: 1 page if <10 years experience, max 2 pages otherwise.
- Consider a **professional summary or objective** at the top.
""")

    st.markdown("### 🤖 Applicant Tracking System (ATS) Optimization Tips")
    st.markdown("""
- Use **standard section headings** (Education, Experience, Skills, Projects).
- Include **keywords** from the job post in your skills and experience.
- Avoid complex layouts: no tables, headers, footers, or graphics.
- Stick to **common fonts**: Arial, Calibri, Times New Roman, Verdana (10–12pt).
- Use **reverse chronological order** for experience and education.
- Save as **PDF or DOCX** (check the application instructions).
""")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Smart Job Matcher | Designed to help you find your perfect career match</p>
</div>
""", unsafe_allow_html=True)
