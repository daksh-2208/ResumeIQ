import os
import re
import pdfplumber
import nltk
from flask import Flask, request, jsonify, render_template
from collections import Counter

# ==========================================
# NLTK SETUP
# ==========================================

nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

app = Flask(__name__)

STOP_WORDS = set(stopwords.words('english'))

# ==========================================
# ROLE PROFILES
# ==========================================

ROLE_PROFILES = {
    "AI/ML Engineer": {
        "skills": {
            "python": 10,
            "machine learning": 10,
            "tensorflow": 9,
            "pytorch": 9,
            "deep learning": 9,
            "nlp": 8,
            "pandas": 7,
            "numpy": 7,
            "scikit-learn": 8,
            "keras": 7,
            "sql": 6,
            "aws": 6,
            "gradio": 5,
            "git": 4
        }
    },

    "Backend Developer": {
        "skills": {
            "java": 10,
            "spring boot": 10,
            "rest api": 9,
            "mongodb": 8,
            "mysql": 8,
            "maven": 7,
            "docker": 7,
            "aws": 6,
            "sql": 8,
            "git": 5,
            "microservices": 7,
            "authentication": 7,
            "crud": 7,
            "python": 5
        }
    },

    "Frontend Developer": {
        "skills": {
            "react": 10,
            "javascript": 10,
            "html": 8,
            "css": 8,
            "bootstrap": 7,
            "angular": 8,
            "vue": 7,
            "ui": 6,
            "responsive": 6
        }
    },

    "Data Analyst": {
        "skills": {
            "sql": 10,
            "python": 9,
            "pandas": 8,
            "numpy": 8,
            "matplotlib": 7,
            "excel": 7,
            "tableau": 8,
            "power bi": 8,
            "statistics": 7,
            "analytics": 7
        }
    },

    "DevOps/Cloud": {
        "skills": {
            "docker": 10,
            "aws": 10,
            "azure": 9,
            "kubernetes": 10,
            "ci/cd": 8,
            "linux": 7,
            "terraform": 8,
            "jenkins": 7,
            "git": 6
        }
    }
}

# ==========================================
# SYNONYMS
# ==========================================

SYNONYM_MAP = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "restful api": "rest api",
    "rest apis": "rest api",
    "apis": "rest api",
    "api": "rest api",
    "springboot": "spring boot",
    "spring-boot": "spring boot",
    "nodejs": "node.js",
    "node-js": "node.js",
    "js": "javascript",
    "github": "git",
    "sklearn": "scikit-learn"
}

# ==========================================
# RESUME ANALYZER
# ==========================================

class ResumeAnalyzer:

    def __init__(self, resume_text, jd_text):

        self.raw_resume = resume_text
        self.raw_jd = jd_text

        self.resume_clean = self.normalize_text(resume_text)
        self.jd_clean = self.normalize_text(jd_text)

        self.jd_role = self.predict_role()

    # ======================================
    # NORMALIZATION
    # ======================================

    def normalize_text(self, text):

        text = text.lower()

        text = re.sub(r'[^\w\s]', ' ', text)

        text = re.sub(r'[-/]', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()

        for key, value in SYNONYM_MAP.items():
            text = text.replace(key, value)

        return text

    # ======================================
    # ROLE DETECTION
    # ======================================

    def predict_role(self):

        role_scores = {}

        for role, data in ROLE_PROFILES.items():

            score = 0

            for skill, weight in data["skills"].items():

                if f' {skill} ' in f' {self.jd_clean} ':
                    score += weight

            role_scores[role] = score

        best_role = max(role_scores, key=role_scores.get)

        return best_role

    # ======================================
    # SKILL EXTRACTION
    # ======================================

    def extract_skills(self, text):

        clean_text = self.normalize_text(text)

        found = set()

        all_skills = set()

        for role in ROLE_PROFILES.values():
            all_skills.update(role["skills"].keys())

        for skill in all_skills:

            if f' {skill} ' in f' {clean_text} ':
                found.add(skill)

        return sorted(list(found))

    # ======================================
    # KEYWORD STUFFING DETECTION
    # ======================================

    def detect_stuffing(self):

        words = self.resume_clean.split()

        total_words = len(words)

        if total_words == 0:
            return 0

        counts = Counter(words)

        penalty = 0

        for word, count in counts.items():

            if (
                word not in STOP_WORDS and
                len(word) > 2 and
                (count / total_words) > 0.07
            ):
                penalty += 10

        return min(penalty, 30)

    # ======================================
    # ROLE MATCH SCORE
    # ======================================

    def calculate_role_scores(self):

        scores = {}

        for role, data in ROLE_PROFILES.items():

            obtained = 0
            total = sum(data["skills"].values())

            for skill, weight in data["skills"].items():

                if f' {skill} ' in f' {self.resume_clean} ':
                    obtained += weight

            final_score = int((obtained / total) * 100)

            scores[role] = min(final_score, 95)

        return scores

    # ======================================
    # PROJECT SCORE
    # ======================================

    def calculate_project_score(self):

        backend_keywords = [
            "spring boot",
            "mongodb",
            "mysql",
            "rest api",
            "authentication",
            "crud",
            "microservices",
            "docker",
            "aws",
            "maven"
        ]

        ai_keywords = [
            "tensorflow",
            "pytorch",
            "machine learning",
            "deep learning",
            "nlp",
            "keras",
            "scikit-learn",
            "prediction"
        ]

        frontend_keywords = [
            "react",
            "javascript",
            "html",
            "css",
            "bootstrap",
            "responsive"
        ]

        data_keywords = [
            "analytics",
            "tableau",
            "power bi",
            "dashboard",
            "statistics",
            "matplotlib"
        ]

        all_keywords = (
            backend_keywords +
            ai_keywords +
            frontend_keywords +
            data_keywords
        )

        matched = []

        for kw in all_keywords:

            if f' {kw} ' in f' {self.resume_clean} ':
                matched.append(kw)

        score = min(int((len(matched) / 12) * 100), 95)

        return score, matched

    # ======================================
    # COMPLETENESS SCORE
    # ======================================

    def completeness_score(self):

        score = 100

        text = self.raw_resume.lower()

        if "linkedin" not in text:
            score -= 10

        if "github" not in text:
            score -= 5

        if "projects" not in text:
            score -= 15

        if "skills" not in text:
            score -= 10

        if "education" not in text:
            score -= 10

        if "internship" not in text and "experience" not in text:
            score -= 15

        return max(score, 0)

    # ======================================
    # CERTIFICATION SCORE
    # ======================================

    def certification_score(self):

        text = self.raw_resume.lower()

        if any(x in text for x in [
            "certification",
            "certificate",
            "course",
            "udemy",
            "coursera",
            "databricks"
        ]):
            return 80

        return 0

    # ======================================
    # MAIN ATS CALCULATION
    # ======================================

    def calculate_scores(self):

        resume_skills = self.extract_skills(self.raw_resume)

        jd_skills = self.extract_skills(self.raw_jd)

        # -------------------------------
        # SKILL MATCH
        # -------------------------------

        role_data = ROLE_PROFILES.get(
            self.jd_role,
            ROLE_PROFILES["Backend Developer"]
        )

        total_weight = 0
        matched_weight = 0

        for skill, weight in role_data["skills"].items():

            if skill in jd_skills:
                total_weight += weight

                if skill in resume_skills:
                    matched_weight += weight

        skill_score = 0

        if total_weight > 0:
            skill_score = int(
                (matched_weight / total_weight) * 100
            )

        # -------------------------------
        # PROJECT SCORE
        # -------------------------------

        project_score, matched_techs = self.calculate_project_score()

        # -------------------------------
        # COMPLETENESS
        # -------------------------------

        completeness = self.completeness_score()

        # -------------------------------
        # CERTIFICATIONS
        # -------------------------------

        cert_score = self.certification_score()

        # -------------------------------
        # ATS SCORE
        # -------------------------------

        raw_ats = (
            (0.50 * skill_score) +
            (0.20 * project_score) +
            (0.20 * completeness) +
            (0.10 * cert_score)
        )

        stuffing_penalty = self.detect_stuffing()

        final_ats = max(
            0,
            int(raw_ats - stuffing_penalty)
        )

        # -------------------------------
        # MISSING SKILLS
        # -------------------------------

        missing_skills = []

        for skill in jd_skills:

            if skill not in resume_skills:
                missing_skills.append(skill)

        # -------------------------------
        # ROLE SCORES
        # -------------------------------

        role_match_scores = self.calculate_role_scores()

        # -------------------------------
        # PROJECT REASON
        # -------------------------------

        if project_score >= 75:
            reason = f"Resume demonstrates strong {self.jd_role} implementation experience."

        elif project_score >= 45:
            reason = f"Resume shows moderate {self.jd_role} project exposure."

        elif project_score > 0:
            reason = f"Resume has limited {self.jd_role} project relevance."

        else:
            reason = "No strong project relevance detected."

        return {
            "ats_score": final_ats,

            "multi_scores": {
                "skill_match": skill_score,
                "project_match": project_score,
                "completeness": completeness,
                "certifications": cert_score
            },

            "project_details": {
                "matched_techs": matched_techs,
                "reason": reason
            },

            "skills": resume_skills,

            "missing_skills": missing_skills,

            "domain": self.jd_role,

            "role_match_scores": role_match_scores
        }

    # ======================================
    # INSIGHTS
    # ======================================

    def generate_insights(self, scores):

        strengths = []
        weaknesses = []
        recommendations = []

        resume_skills = scores["skills"]

        # -------------------------------
        # STRENGTHS
        # -------------------------------

        if scores["multi_scores"]["skill_match"] > 70:
            strengths.append(
                "Strong alignment with job technical requirements."
            )

        backend_stack = [
            "spring boot",
            "mongodb",
            "mysql",
            "rest api"
        ]

        backend_matches = [
            x for x in backend_stack
            if x in resume_skills
        ]

        if len(backend_matches) >= 3:
            strengths.append(
                "Demonstrates strong backend development stack knowledge."
            )

        ai_stack = [
            "tensorflow",
            "machine learning",
            "scikit-learn",
            "deep learning"
        ]

        ai_matches = [
            x for x in ai_stack
            if x in resume_skills
        ]

        if len(ai_matches) >= 2:
            strengths.append(
                "Demonstrates practical AI/ML experience."
            )

        if "internship" in self.resume_clean:
            strengths.append(
                "Has internship or practical experience exposure."
            )

        if scores["multi_scores"]["certifications"] > 0:
            strengths.append(
                "Holds industry certifications."
            )

        # -------------------------------
        # WEAKNESSES
        # -------------------------------

        if len(scores["missing_skills"]) > 0:

            weaknesses.append(
                "Missing important skills: " +
                ", ".join(scores["missing_skills"][:4])
            )

        if "docker" not in resume_skills:
            weaknesses.append(
                "Limited deployment/cloud exposure."
            )

        if "projects" not in self.resume_clean:
            weaknesses.append(
                "Project section appears weak or missing."
            )

        if len(weaknesses) == 0:
            weaknesses.append(
                "No major weaknesses detected for selected role."
            )

        # -------------------------------
        # RECOMMENDATIONS
        # -------------------------------

        if "docker" not in resume_skills:
            recommendations.append(
                "Add Docker and cloud deployment projects."
            )

        if "aws" not in resume_skills:
            recommendations.append(
                "Add AWS deployment or cloud certification."
            )

        if "react" not in resume_skills:
            recommendations.append(
                "Build a modern frontend project using React."
            )

        if len(recommendations) == 0:
            recommendations.append(
                "Continue building advanced projects."
            )

        # -------------------------------
        # TOP ROLE RECOMMENDATIONS
        # -------------------------------

        sorted_roles = sorted(
            scores["role_match_scores"],
            key=scores["role_match_scores"].get,
            reverse=True
        )

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "recommended_roles": sorted_roles[:4]
        }

# ==========================================
# PDF EXTRACTION
# ==========================================

def extract_text_from_pdf(file_storage):

    text = ""

    try:

        with pdfplumber.open(file_storage) as pdf:

            for page in pdf.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + " "

    except Exception:
        return ""

    return text.strip()

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():

    if 'resume' not in request.files:
        return jsonify({
            "error": "Resume file missing"
        }), 400

    if 'job_description' not in request.form:
        return jsonify({
            "error": "Job description missing"
        }), 400

    file = request.files['resume']

    jd_text = request.form['job_description']

    if file.filename == '':
        return jsonify({
            "error": "No file selected"
        }), 400

    resume_text = extract_text_from_pdf(file)

    if not resume_text:
        return jsonify({
            "error": "Could not extract text from PDF"
        }), 400

    analyzer = ResumeAnalyzer(
        resume_text,
        jd_text
    )

    scores = analyzer.calculate_scores()

    insights = analyzer.generate_insights(scores)

    return jsonify({
        **scores,
        "insights": insights,
        "stats": {
            "word_count": len(resume_text.split())
        }
    })

# ==========================================
# RUN
# ==========================================

if __name__ == '__main__':

    app.run(
        debug=True,
        port=5000
    )