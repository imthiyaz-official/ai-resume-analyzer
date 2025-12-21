import streamlit as st
import pickle
import re
import docx
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load trained objects
# =========================
svc_model = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))

# =========================
# Skills Database
# =========================
SKILLS_DB = {
    "Programming": ["python", "java", "c", "c++", "sql"],
    "Data Science": ["data analysis", "machine learning", "deep learning", "statistics", "nlp", "computer vision"],
    "Libraries": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
    "Visualization": ["matplotlib", "seaborn", "power bi", "tableau"],
    "Cloud & DevOps": ["aws", "docker", "kubernetes", "jenkins", "ci/cd", "terraform", "linux"],
    "Tools": ["git", "github", "jupyter", "colab", "vscode", "postman"],
    "Java Developer": ["spring", "spring boot", "hibernate", "jsp", "servlets", "maven", "jdbc"],
    "Backend": ["django", "flask", "rest api", "microservices"],
    "Full Stack": ["html", "css", "javascript", "react", "node"],
    "Testing": ["selenium", "testng", "automation testing"],
    "Database": ["mysql", "postgresql", "mongodb", "oracle", "redis",
    "sql server", "dynamodb", "sqlite","cassandra","firebase","elasticsearch"]
}
# =========================
# Text Cleaning
# =========================
def cleanResume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"#\S+|@\S+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# =========================
# File Readers
# =========================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except:
        return file.read().decode("latin-1")
def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == "docx":
        return extract_text_from_docx(uploaded_file)
    elif ext == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file format")
# =========================
# Prediction
# =========================
def predict_category(resume_text):
    vec = tfidf.transform([cleanResume(resume_text)]).toarray()
    pred = svc_model.predict(vec)
    return le.inverse_transform(pred)[0]
def category_confidence(resume_text):
    try:
        vec = tfidf.transform([cleanResume(resume_text)]).toarray()
        probs = svc_model.predict_proba(vec)[0]
        idx = np.argmax(probs)
        return le.inverse_transform([idx])[0], round(probs[idx] * 100, 2)
    except:
        return None, None
# =========================
# Skills & Scoring
# =========================
def extract_skills(text):
    text = text.lower()
    return {k: [s for s in v if s in text] for k, v in SKILLS_DB.items() if any(s in text for s in v)}
def resume_strength_score(text, skills):
    score = min(sum(len(v) for v in skills.values()) * 4, 40)
    if "project" in text: score += 20
    if "intern" in text or "experience" in text: score += 20
    if "certificate" in text: score += 20
    if "leadership" in text or "volunteer" in text: score += 10
    return min(score, 100)
# =========================
# Charts
# =========================
def plot_skill_pie(skills):
    labels = list(skills.keys())
    sizes = [len(v) for v in skills.values()]
    if not sizes:
        return None
    fig, ax = plt.subplots(figsize=(10,4))  # 🔽 reduced size

    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)

    ax.axis("equal")
    return fig

def radar_chart(skills):
    categories = list(SKILLS_DB.keys())
    values = [len(skills.get(cat, [])) for cat in categories]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(categories)+1)
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_thetagrids(angles[:-1]*180/np.pi, categories, fontsize=8)
    ax.set_title("Skill Balance Radar")
    ax.set_rlim(0, max(5, max(values)))

    return fig

# =========================
# ATS Score
# =========================
def ats_score(resume_text, jd_text):
    r_vec = tfidf.transform([cleanResume(resume_text)])
    jd_vec = tfidf.transform([cleanResume(jd_text)])
    return round(cosine_similarity(r_vec, jd_vec)[0][0] * 100, 2)


# =========================
# Suggestions
# =========================
def resume_suggestions(score, skills):
    tips = []
    if score < 60: tips.append("Add more projects and technical depth.")
    if "Cloud & DevOps" not in skills: tips.append("Add basic cloud/DevOps exposure.")
    if "Visualization" not in skills: tips.append("Mention visualization tools.")
    if not tips: tips.append("Resume looks strong. Start tailoring to job roles.")
    return tips

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config("Resume Analyzer", "📄", layout="wide")
    st.title("📄 AI Resume Analyzer Dashboard")


    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])

    if uploaded_file:
        resume_text = handle_file_upload(uploaded_file)
        st.success("Resume processed successfully!")
        st.subheader("📝 Resume Text Preview")
        st.text_area("Resume Text", resume_text, height=200)

        category = predict_category(resume_text)
        st.subheader("✅ Predicted Job Category")
        st.markdown(f"### **{category}**")

        cat, conf = category_confidence(resume_text)
        if conf:
            st.metric("📈 Category Confidence", f"{conf}%")

        skills = extract_skills(resume_text)

        st.subheader("🛠️ Extracted Skills")
        for k, v in skills.items():
            st.write(f"**{k}:** {', '.join(v)}")

        st.subheader("📊 Skill Distribution")
        fig = plot_skill_pie(skills)
        if fig: st.pyplot(fig)

        st.subheader("🕸️ Skill Balance")
        st.pyplot(radar_chart(skills))

        score = resume_strength_score(resume_text, skills)
        st.subheader("🧠 Resume Strength Score")
        st.progress(score/100)
        st.metric("Score", f"{score}/100")


        st.subheader("🤖 Resume Suggestions")
        for tip in resume_suggestions(score, skills):
            st.write("•", tip)

        st.subheader("📄 ATS Match Score")
        jd = st.text_area("Paste Job Description")
        if jd:
            ats = ats_score(resume_text, jd)
            st.progress(ats/100)
            st.metric("ATS Match", f"{ats}%")


if __name__ == "__main__":
    main()
