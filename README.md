# Smart Job Matcher

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit\&logoColor=white)
![Language](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered resume matcher, analyzer, and job market explorer that helps job seekers find the most relevant job listings based on their resume, and provides feedback to improve their resumes. Built with bilingual support (English and Mongolian), this app uses NLP, semantic similarity, and data visualization to create a smarter job search experience.


---

## Features

### ✉ Resume-to-Job Matching

* Upload your resume (PDF or DOCX)
* Analyze the structure and extract skills
* Filter jobs by sector and salary
* Perform semantic matching with job listings using SentenceTransformer
* Visualize match scores with charts
* Highlight matched and missing skills
* Download matched jobs as Excel

###  Resume Analysis

* Parse resume sections (Contact, Education, Skills, etc.)
* Detect email, phone, LinkedIn, and more
* Compute completeness score
* Visualize top keywords with a word cloud
* Generate suggestions to improve your resume

###  Job Market Explorer

* Filter jobs by company, sector, and salary
* Visualize:

  * Job distribution by sector (pie chart)
  * Top hiring companies (bar chart)
  * In-demand keywords (word cloud)
* Scrollable job listing with live links

###  Resume Creator

* Fill out a simple form to generate a professional resume
* Download as a .docx file

---

## How It Works

### Skill Extraction

* Uses spaCy and NLTK for extracting domain-specific terms from resume

### Semantic Matching

* Embeds resume and job descriptions using SentenceTransformer
* Calculates cosine similarity to rank jobs by relevance

### Job Sector Filtering

* Jobs are filtered based on predefined keyword groups (English & Mongolian)
* Supports sectors like Education, Tech, Healthcare, Logistics, etc.

### Word Cloud Visualization

* Extracts top keywords from resume and job descriptions
* Displays a word cloud to visualize key focus areas

---

## Tech Stack

| Component          | Technology                    |
| ------------------ | ----------------------------- |
| Web App            | Streamlit                     |
| Resume Parsing     | PyMuPDF, python-docx          |
| Skill Extraction   | spaCy, NLTK                   |
| Semantic Matching  | SentenceTransformers, sklearn |
| Data Visualization | Plotly, Matplotlib, WordCloud |
| File Output        | openpyxl, io                  |

##  Project Structure

For full documentation of the codebase and architecture, see [docs/structure.md](docs/structure.md).

---

## Folder Structure

```
smart-job-matcher/
├── app.py                      # Main Streamlit app
├── resume_parser.py           # Resume parsing functions
├── semantic_matcher.py        # Skill extraction + matching logic
├── requirements.txt           # Python dependencies
└── data/
    └── zangia_filtered_jobs.csv  # Scraped job listing data
```

---

## Installation

```bash
# Clone the repo
$ git clone https://github.com/yourname/smart-job-matcher.git
$ cd smart-job-matcher

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ streamlit run app.py
```

---

## Data Source

* Job data scraped from **Zangia.mn** using BeautifulSoup
* Resume inputs parsed locally from PDF/DOCX

---

## Author

Munkhbold Nyamdorj
American University of Mongolia

---

## License

This project is for educational and demonstration purposes.

---

