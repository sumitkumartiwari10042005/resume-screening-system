# AI-Powered Resume Screening & Ranking System

A Streamlit app that ranks PDF resumes against a job description. It scores each candidate on skills match and text similarity, then shows which skills they have and which they are missing.


## Features

- Paste a job description and upload multiple PDF resumes at once
- Blended 0-100 score: **60% skills match + 40% semantic similarity**
- Automatic detection of **required** and **good-to-have** skills from the job description
- Per-candidate breakdown: matched skills, missing required skills, missing good-to-have skills
- Match labels: **Strong** (70+), **Moderate** (40-69), **Low** (below 40)
- Summary stats: candidate count, top score, average, gap between #1 and #2
- Export the ranking as CSV
- Handles scanned or unreadable PDFs with a clear warning instead of crashing
- Dark, card-based UI

## How scoring works

| Component | Weight | Method |
|---|---|---|
| Skills match | 60% | Skills are detected in the job description from a built-in dictionary of ~45 skills with aliases (e.g. `k8s` = Kubernetes). Required skills count 2x, good-to-have skills 1x. Each resume is checked for the same skills using whole-word matching. |
| Semantic similarity | 40% | TF-IDF (unigrams + bigrams) cosine similarity between the job description and the resume, rescaled to 0-100. |

The JD is split into sections by headings such as "Required Skills" and "Good to Have". Skills in a good-to-have section count once, and everything else counts as required. If no known skills are found in the JD, the score falls back to semantic similarity only.

**Sample result** (backend SDE-1 job description, one strong and one weak resume):

| Candidate | Score | Verdict |
|---|---|---|
| Strong backend resume (Java, Spring Boot, JWT, Docker, Redis, Kafka) | 72.2 | Strong match |
| Web design resume (HTML, CSS, WordPress) | 7.3 | Low match |

## Tech stack

- Python 3.10+
- [Streamlit](https://streamlit.io/) for the UI
- [scikit-learn](https://scikit-learn.org/) for TF-IDF and cosine similarity
- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF text extraction
- pandas for the CSV export

## Project structure

```
.
├── .streamlit/
│   └── config.toml        # dark theme settings
├── Main.py                # Streamlit UI and app flow
├── Scoring.py             # skill extraction, scoring, ranking
├── Styles.css             # custom styling
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/sumitkumartiwari10042005/resume-screening-system.git
cd AI-powered-Resume-Screening-and-Ranking-System
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

Run this from the project root so the `.streamlit/config.toml` theme is picked up:

```bash
streamlit run Main.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Paste the full job description. Using headings like **Required Skills** and **Good to Have** gives the most accurate skill weighting.
2. Upload one or more PDF resumes.
3. Click **Analyse**.
4. Review the ranked candidate cards and download the CSV if needed.

## Limitations

- Only skills in the built-in dictionary are detected. To add more, edit the `SKILLS` dictionary in `Scoring.py`.
- Scanned (image-only) PDFs have no extractable text and are skipped. OCR is not included.
- TF-IDF measures word overlap, not meaning, so it can miss synonyms the skills dictionary doesn't cover.
- Scores are a screening aid and should not replace human review.

## Roadmap

- Sentence-embedding similarity instead of TF-IDF
- OCR support for scanned resumes
- Experience and education scoring
- Support for DOCX resumes
- Editable skills dictionary from the UI
