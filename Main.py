from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from Scoring import analyse_resumes

st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'Styles.css').read_text()}</style>", unsafe_allow_html=True)

BAND_COLOR = {"Strong match": "#4FD1A5", "Moderate match": "#F2C46D", "Low match": "#FF8A94"}


def extract_text_from_pdf(file) -> str:
    pdf = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)


def step(n: int, title: str) -> str:
    return f'<div class="rr-step"><span class="rr-num">{n}</span><span>{title}</span></div>'


def chips(items, cls: str) -> str:
    return "".join(f'<span class="rr-chip {cls}">{escape(i)}</span>' for i in items)


def ring(score: float, color: str) -> str:
    dash = 213.6 * max(0.0, min(score, 100.0)) / 100
    return (
        f'<svg width="96" height="96" viewBox="0 0 80 80" role="img" aria-label="Score {score:.1f} out of 100">'
        '<circle cx="40" cy="40" r="34" fill="none" stroke="#262D38" stroke-width="8"/>'
        f'<circle cx="40" cy="40" r="34" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.1f} 213.6" transform="rotate(-90 40 40)"/></svg>'
    )


def candidate_card(rank: int, r: dict) -> str:
    color = BAND_COLOR[r["label"]]
    matched = chips(r["matched_skills"], "ok") or '<span class="rr-file">None found</span>'
    missing = chips(r["missing_skills"], "miss")
    nice = chips(r["missing_nice_to_have"], "nice")
    groups = f'<div class="rr-group"><div class="rr-lbl">Matched skills</div>{matched}</div>'
    if missing:
        groups += f'<div class="rr-group"><div class="rr-lbl">Missing (required)</div>{missing}</div>'
    if nice:
        groups += f'<div class="rr-group"><div class="rr-lbl">Missing (good to have)</div>{nice}</div>'
    return (
        '<div class="rr-card rr-cand">'
        f'<div class="rr-ring">{ring(r["score"], color)}<div class="rr-ring-n">{r["score"]:.1f}</div></div>'
        '<div class="rr-body">'
        f'<div class="rr-head"><span class="rr-rank{" top" if rank == 1 else ""}">#{rank}</span>'
        f'<span class="rr-name">{escape(r["name"])}</span><span class="rr-file">{escape(r["file"])}</span></div>'
        f'<div class="rr-verdict" style="color:{color}">{r["label"]}</div>{groups}</div></div>'
    )


def stat(label: str, value: str) -> str:
    return f'<div class="rr-card rr-stat"><div class="rr-lbl">{label}</div><div class="rr-val">{value}</div></div>'


st.markdown(
    '<div class="rr-top"><div class="rr-brand"><div class="rr-logo">'
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#06231B" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3h8l4 4v14H6z"/><path d="M9 13l2 2 4-4"/></svg>'
    '</div>ResumeRank</div><span class="rr-tag">Resume screening &amp; candidate ranking</span></div>'
    '<h1 class="rr-h1">Rank candidates against a job description</h1>'
    '<p class="rr-sub">Paste the role, drop in resumes, and see who fits best along with the skills each candidate is missing.</p>',
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, key="jd_card"):
        st.markdown(step(1, "Job description"), unsafe_allow_html=True)
        job_description = st.text_area("Paste the full description", height=220, key="jd")
        st.caption(f"{len(job_description):,} characters · Required and good-to-have skills are detected automatically")
with right:
    with st.container(border=True, key="up_card"):
        st.markdown(step(2, "Resumes"), unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)
        n = len(uploaded_files or [])
        run = st.button(
            f"Analyse {n} resume{'s' if n != 1 else ''}" if n else "Analyse resumes",
            type="primary",
            disabled=not (job_description.strip() and n),
        )

if run:
    with st.spinner("Analysing resumes..."):
        docs, skipped = [], []
        for f in uploaded_files:
            try:
                text = extract_text_from_pdf(f)
            except Exception:
                skipped.append((f.name, "could not be read as a PDF"))
                continue
            if len(text.strip()) < 30:
                skipped.append((f.name, "has no extractable text (it may be a scanned image)"))
                continue
            docs.append((f.name, text))
        st.session_state["results"] = analyse_resumes(job_description, docs) if docs else []
        st.session_state["skipped"] = skipped

st.markdown('<div style="height:20px"></div>' + step(3, "Ranking"), unsafe_allow_html=True)

for name, reason in st.session_state.get("skipped", []):
    st.warning(f"**{name}** {reason}, so it was left out of the ranking.")

results = st.session_state.get("results")
if results:
    scores = [r["score"] for r in results]
    gap = f"{scores[0] - scores[1]:.1f}" if len(scores) > 1 else "—"
    st.markdown(
        '<div class="rr-stats">'
        + stat("Candidates", str(len(results)))
        + stat("Top score", f"{scores[0]:.1f}")
        + stat("Average", f"{sum(scores) / len(scores):.1f}")
        + stat("Gap to #2", gap)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("".join(candidate_card(i, r) for i, r in enumerate(results, 1)), unsafe_allow_html=True)

    df = pd.DataFrame(
        {
            "Rank": range(1, len(results) + 1),
            "Candidate": [r["name"] for r in results],
            "File": [r["file"] for r in results],
            "Score": scores,
            "Verdict": [r["label"] for r in results],
            "Matched skills": [", ".join(r["matched_skills"]) for r in results],
            "Missing required skills": [", ".join(r["missing_skills"]) for r in results],
        }
    )
    st.download_button("Export results as CSV", df.to_csv(index=False).encode("utf-8"), "resume_rankings.csv", "text/csv")
elif results == []:
    st.markdown('<div class="rr-card rr-empty">None of the uploaded resumes could be analysed.</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="rr-card rr-empty">Add a job description and at least one PDF resume,<br>then press <b>Analyse</b> to see the ranking.</div>',
        unsafe_allow_html=True,
    )