##  Smart Job Matcher — Detailed Component & Function Overview

This document breaks down the architecture and logic of the Smart Job Matcher app, explaining each module, function, and interaction in detail.

---

###  Main Application Modules

| **Component**       | **File**                        | **Function / Feature**                     | **Purpose**                                                                  |
| ------------------- | ------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------- |
| `app.py`            | `app.py`                        | Main entry point of the Streamlit app      | Handles UI layout, mode selection, and dynamic user interaction              |
| Resume Upload       | `app.py`                        | File uploader (`st.file_uploader`)         | Accepts PDF or DOCX resumes, triggers text extraction and session updates    |
| Resume Creator      | `app.py`                        | Interactive form to input resume data      | Generates downloadable `.docx` resume using `python-docx`                    |
| Resume Analysis     | `app.py`                        | Skill & structure analysis from resume     | Shows resume completeness, extracted keywords, contact info, and tips        |
| Job Matching        | `app.py`                        | Semantic similarity computation            | Compares resume text to jobs using transformer embeddings and skill matching |
| Job Market Explorer | `app.py`                        | Visualization of job dataset               | Allows filtering jobs and visualizing insights with charts and word clouds   |
| Visualizations      | `app.py`                        | Plotly, WordCloud, matplotlib              | Used to build match score bars, pie charts, word clouds, etc.                |
| Job Data            | `data/zangia_filtered_jobs.csv` | Scraped and cleaned dataset from Zangia.mn | Used as the primary source for all job matching and exploration              |

#### Example Code: Resume Upload Snippet

```python
uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(uploaded_file)
```

---

###  `resume_parser.py` — Resume Parsing Functions

| **Function**                | **Purpose**                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `extract_text_from_pdf()`   | Uses PyMuPDF (`fitz`) to extract structured text block by block from PDF resumes              |
| `extract_text_from_docx()`  | Uses `docx2txt` and `python-docx` to parse headers, paragraphs, and tables from DOCX resumes  |
| `clean_resume_text()`       | Applies regex to clean extra whitespace, normalize bullet points, and structure paragraphs    |
| `extract_resume_sections()` | Detects resume sections (contact, education, skills, etc.) based on heading patterns          |
| `extract_contact_info()`    | Uses regex to identify email, phone numbers, LinkedIn URLs, location, and possible name lines |
| `analyze_resume()`          | Scores completeness based on presence and length of key sections and extracted contact info   |

#### Example Code: Clean Resume Text

```python
def clean_resume_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.replace('•', '\n• ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

---

###  `semantic_matcher.py` — NLP Matching Functions

| **Function**                   | **Purpose**                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `preprocess_text()`            | Converts text to lowercase, removes special characters, collapses whitespace                   |
| `extract_resume_keywords()`    | Uses NLTK, RegexpTokenizer, and spaCy to extract top keywords from resumes                     |
| `extract_skills_from_resume()` | Searches text for known tech, soft, and Mongolian job skills plus NER-extracted entities       |
| `semantic_match_resume()`      | Embeds resume and jobs with `SentenceTransformer`; ranks jobs by cosine similarity             |
| `get_skill_matches()`          | Compares extracted resume skills with keywords and skills in each job; highlights missing ones |

#### Example Code: Semantic Matching

```python
def semantic_match_resume(resume_text, jobs_df, top_n=10):
    resume_embedding = model.encode([preprocess_text(resume_text)])[0]
    jobs_df["combined_text"] = (
        jobs_df["Job title"].astype(str) + " " +
        jobs_df["Job description"].astype(str) + " " +
        jobs_df["Requirements"].astype(str)
    ).apply(preprocess_text)
    job_embeddings = model.encode(jobs_df["combined_text"].tolist())
    similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0]
    jobs_df["match_score"] = similarity_scores * 100
    return jobs_df.sort_values(by="match_score", ascending=False).head(top_n)
```

---

###  Data Flow Summary

1. User uploads resume → parsed by `resume_parser.py`
2. Resume text analyzed → structure and skills stored in session state
3. User switches to matching mode → `semantic_matcher.py` computes top matches
4. Matching results visualized → score bars, matched/missing skills shown
5. Resume analyzed → feedback given in the form of score, word cloud, and improvement tips
6. Explorer mode → displays overall job market trends (sectors, companies, keyword cloud)

---

For visual architecture, refer to the `docs/architecture.png` file in the repository. This diagram outlines relationships between modules, app logic, and data dependencies.

---

*Last updated: May 2025*
