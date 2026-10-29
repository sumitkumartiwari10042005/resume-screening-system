from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS_WEIGHT = 0.6
SEMANTIC_WEIGHT = 0.4
REQUIRED_WEIGHT = 2
GOOD_WEIGHT = 1
SEMANTIC_REF = 0.35

SKILLS: dict[str, list[str]] = {
    "Java": ["java"],
    "Spring Boot": ["spring boot", "springboot", "spring-boot"],
    "Hibernate/JPA": ["hibernate", "jpa"],
    "REST APIs": ["rest api", "restful", "rest-api", "rest web service"],
    "Microservices": ["microservice", "micro-service", "microservices architecture"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sqlite", "oracle db", "t-sql", "pl/sql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo db"],
    "Redis": ["redis"],
    "Kafka": ["kafka"],
    "RabbitMQ": ["rabbitmq"],
    "Docker": ["docker", "containeriz", "dockeriz"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services", "ec2"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Terraform": ["terraform"],
    "Git": ["git", "version control"],
    "GitHub Actions": ["github actions"],
    "Jenkins": ["jenkins"],
    "CI/CD": ["ci/cd", "cicd", "ci cd", "continuous integration", "continuous delivery", "github actions", "jenkins", "gitlab ci"],
    "JWT": ["jwt", "json web token"],
    "OAuth2": ["oauth2", "oauth 2.0", "oauth"],
    "Unit Testing": ["unit test", "integration test", "unit and integration test", "junit", "mockito", "pytest", "jest"],
    "Data Structures & Algorithms": ["data structures", "algorithms", "dsa"],
    "System Design": ["system design", "low level design", "high level design"],
    "Python": ["python"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "React": ["react", "reactjs", "react.js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Angular": ["angular"],
    "HTML/CSS": ["html", "css"],
    "C++": ["c++"],
    "C#": ["c#"],
    "Go": ["golang"],
    "Linux": ["linux"],
    "Machine Learning": ["machine learning"],
    "NLP": ["nlp", "natural language processing"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Pandas": ["pandas"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Agile": ["agile", "scrum"],
}


def _compile(aliases: list[str]) -> re.Pattern:
    body = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9+#])(?:{body})(?:s|ed|ing|ation)?(?![a-z0-9+#])", re.I)


_PATTERNS = {name: _compile(al) for name, al in SKILLS.items()}


def _found(text: str) -> set[str]:
    text = re.sub(r"\s+", " ", text or "")
    return {name for name, pat in _PATTERNS.items() if pat.search(text)}


_GOOD_HDR = re.compile(r"good to have|nice to have|preferred|bonus|plus|desirable|added advantage", re.I)
_REQ_HDR = re.compile(r"required|must have|mandatory|requirements|essential|key skills|technical skills|core skills", re.I)
_OTHER_HDR = re.compile(r"responsibilit|about|what you|qualification|benefit|perks|overview|who you|location|education|how to apply", re.I)
_INLINE_GOOD = re.compile(r"\b(?:good to have|nice to have|preferred|bonus)\b[^:\n]{0,20}[:\-\u2013]", re.I)


def _header_kind(line: str) -> str | None:
    if len(line) > 40 or line.endswith(".") or line[0] in "-*\u2022\u2013\u25cf\u25aa":
        return None
    if not (line.endswith(":") or len(line.split()) <= 4):
        return None
    if _GOOD_HDR.search(line):
        return "good"
    if _REQ_HDR.search(line):
        return "required"
    if _OTHER_HDR.search(line):
        return "other"
    return None


def _split_sections(jd: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"required": [], "good": [], "other": []}
    current = "other"
    for raw in jd.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _INLINE_GOOD.search(line)
        if m and line[m.end():].strip():
            sections[current].append(line[: m.start()])
            sections["good"].append(line[m.end():])
            current = "good"
            continue
        kind = _header_kind(line)
        if kind:
            current = kind
            continue
        sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def extract_jd_skills(jd: str) -> dict[str, list[str]]:
    sec = _split_sections(jd)
    req, good, other = _found(sec["required"]), _found(sec["good"]), _found(sec["other"])
    required = req | (other - good)
    nice = good - required
    order = list(SKILLS)
    return {
        "required": [s for s in order if s in required],
        "good": [s for s in order if s in nice],
    }


def extract_name(text: str, filename: str = "") -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if 1 < len(line.split()) <= 4 and len(line) <= 40 and re.fullmatch(r"[A-Za-z.' -]+", line):
            return line.title() if line.isupper() else line
        break
    stem = Path(filename).stem if filename else ""
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b(resume|cv)\b", "", stem, flags=re.I)
    stem = re.sub(r"\b[A-Za-z]\b", "", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "Unknown candidate"

def label_for(score: float) -> str:
    if score >= 70:
        return "Strong match"
    if score >= 40:
        return "Moderate match"
    return "Low match"


def _semantic_scores(jd: str, texts: list[str]) -> list[float]:
    clean = lambda t: re.sub(r"\s+", " ", t or "")
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = vec.fit_transform([clean(jd)] + [clean(t) for t in texts])
    except ValueError:  # empty vocabulary
        return [0.0] * len(texts)
    sims = cosine_similarity(matrix[0], matrix[1:]).flatten()
    return [min(100.0, float(s) / SEMANTIC_REF * 100) for s in sims]


def analyse_resumes(jd: str, resumes: list[tuple[str, str]]) -> list[dict]:
    jd_skills = extract_jd_skills(jd)
    req, good = jd_skills["required"], jd_skills["good"]
    total = REQUIRED_WEIGHT * len(req) + GOOD_WEIGHT * len(good)
    semantic = _semantic_scores(jd, [t for _, t in resumes])

    results = []
    for (filename, text), sem in zip(resumes, semantic):
        have = _found(text)
        got_req = [s for s in req if s in have]
        got_good = [s for s in good if s in have]
        skills_score = None
        if total:
            skills_score = 100 * (REQUIRED_WEIGHT * len(got_req) + GOOD_WEIGHT * len(got_good)) / total
        final = sem if skills_score is None else SKILLS_WEIGHT * skills_score + SEMANTIC_WEIGHT * sem
        final = round(final, 1)
        results.append({
            "file": filename,
            "name": extract_name(text, filename),
            "score": final,
            "skills_score": None if skills_score is None else round(skills_score, 1),
            "semantic_score": round(sem, 1),
            "label": label_for(final),
            "matched_skills": got_req + got_good,
            "missing_skills": [s for s in req if s not in have],
            "missing_nice_to_have": [s for s in good if s not in have],
        })
    return sorted(results, key=lambda r: r["score"], reverse=True)


def score_resume(jd_text: str, resume_text: str, filename: str = "") -> dict:
    return analyse_resumes(jd_text, [(filename, resume_text)])[0]