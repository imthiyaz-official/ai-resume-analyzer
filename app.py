import streamlit as st
import pickle
import re
import docx
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import Circle
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import hashlib
import json
import os
from datetime import datetime
import time
import base64
from PIL import Image as PILImage
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =========================
# Authentication System
# =========================
class AuthSystem:
    def __init__(self):
        self.users_file = "users.json"
        self.load_users()
        self.current_user = None
        self.session_active = False

    def load_users(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
                self.users = self.users if isinstance(self.users, dict) else {}
        else:
            self.users = {}
            self.save_users()

    def save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=4)
            f.flush()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username, password, email=""):
        if username in self.users:
            return False, "Username already exists"

        self.users[username] = {
            'password': self.hash_password(password),
            'email': email,
            'created_at': datetime.now().isoformat(),
            'resumes': []
        }
        self.save_users()
        return True, "Registration successful"

    def login(self, username, password):
        if username not in self.users:
            return False, "Invalid username"

        if self.users[username]['password'] == self.hash_password(password):
            return True, "Login successful"
        return False, "Invalid password"

    def add_resume_to_user(self, username, resume_data):
        if username in self.users:
            if 'resumes' not in self.users[username]:
                self.users[username]['resumes'] = []
            self.users[username]['resumes'].append(resume_data)
            self.save_users()
            return True
        return False

# =========================
# Load trained objects with improved error handling
# =========================
def load_models():
    """Load ML models with fallbacks for cloud deployment"""
    try:
        # Try multiple possible paths for model files
        possible_paths = [
            ("clf.pkl", "tfidf.pkl", "encoder.pkl"),  # Root directory
            ("models/clf.pkl", "models/tfidf.pkl", "models/encoder.pkl"),  # Models subdirectory
            ("/mount/src/ai-resume-analyzer/clf.pkl", "/mount/src/ai-resume-analyzer/tfidf.pkl", "/mount/src/ai-resume-analyzer/encoder.pkl"),  # Streamlit Cloud
        ]
        
        for clf_path, tfidf_path, le_path in possible_paths:
            try:
                if os.path.exists(clf_path):
                    svc_model = pickle.load(open(clf_path, "rb"))
                    tfidf = pickle.load(open(tfidf_path, "rb"))
                    le = pickle.load(open(le_path, "rb"))
                    return svc_model, tfidf, le
            except Exception as e:
                continue
        
        # If no model files found, create dummies
        st.warning("⚠️ Model files not found. Using demo mode with dummy models.")
        return create_dummy_models()
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return create_dummy_models()

def create_dummy_models():
    """Create dummy models for demo purposes"""
    class DummyModel:
        def predict(self, X):
            return ["Software Developer"]
        def decision_function(self, X):
            return np.array([[0.8]])

    class DummyEncoder:
        def inverse_transform(self, X):
            return ["Software Developer"]

    class DummyTfidf:
        def transform(self, X):
            return np.random.rand(1, 100).astype(np.float32)

    svc_model = DummyModel()
    tfidf = DummyTfidf()
    le = DummyEncoder()
    
    return svc_model, tfidf, le

# Load models
svc_model, tfidf, le = load_models()

# Initialize auth system
auth = AuthSystem()

# =========================
# Skills Database
# =========================
SKILLS_DB = {
    "Programming": ["python", "java", "c", "c++", "sql", "javascript", "typescript"],
    "Data Science": ["data analysis", "machine learning", "deep learning", "nlp", "computer vision"],
    "Libraries": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
    "Web Development": ["html", "css", "react", "angular", "vue", "django", "flask", "node.js"],
    "Visualization": ["matplotlib", "seaborn", "power bi", "tableau", "plotly"],
    "Cloud": ["aws", "azure", "gcp", "docker", "kubernetes"],
    "Tools": ["git", "github", "jupyter", "vscode", "jenkins"],
    "Databases": ["mysql", "postgresql", "mongodb", "redis", "sqlite"],
    "DevOps": ["ci/cd", "terraform", "ansible", "puppet"],
    "Soft Skills": ["communication", "teamwork", "problem solving", "leadership", "adaptability"],
    "Project Management": ["agile", "scrum", "kanban", "jira", "confluence"]
}

# =========================
# ROLE-BASED CATEGORY MAPPING
# =========================
ROLE_SKILLS = {
    "Java Developer": ["java", "spring", "spring boot", "hibernate", "microservices", "rest api"],
    "Python Developer": ["python", "django", "flask", "fastapi", "sqlalchemy"],
    "Frontend Developer": ["html", "css", "javascript", "typescript", "react", "angular", "vue"],
    "Backend Developer": ["node.js", "express", "django", "flask", "spring boot", "mongodb", "postgresql"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node.js", "django", "mongodb"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "ci/cd", "aws", "terraform", "ansible"],
    "Data Scientist": ["python", "machine learning", "data analysis", "statistics", "pandas", "numpy"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "data visualization", "statistics"],
    "Machine Learning Engineer": ["machine learning", "deep learning", "tensorflow", "pytorch", "mlops"],
    "Cloud Engineer": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
    "Cybersecurity Analyst": ["network security", "penetration testing", "firewalls", "siem", "cryptography"],
    "Mobile App Developer": ["android", "ios", "react native", "flutter", "kotlin", "swift"],
    "Project Manager": ["project management", "agile", "scrum", "kanban", "jira", "confluence"],
    "QA Engineer": ["testing", "automation", "selenium", "junit", "testng", "cypress"],
    "System Administrator": ["linux", "windows server", "networking", "scripting", "bash", "powershell"]
}

# =========================
# Enhanced Resume Template Engine with PDF Generation
# =========================
class EnhancedResumeTemplateEngine:
    @staticmethod
    def extract_personal_info(resume_text):
        """Extract personal information from resume text"""
        info = {
            "name": "[YOUR NAME]",
            "email": "[Email]",
            "phone": "[Phone]",
            "location": "[City, State]",
            "linkedin": "[LinkedIn URL]",
            "github": "[GitHub URL]",
            "portfolio": "[Portfolio URL]",
            "summary": "[Professional Summary]",
            "objective": "[Career Objective]",
            "skills": "[Key Skills]",
            "experience": "[Work Experience]",
            "education": "[Education Details]",
            "projects": "[Projects]",
            "certifications": "[Certifications]"
        }

        # Extract name (simple heuristic - first line that looks like a name)
        lines = resume_text.split('\n')
        for line in lines[:5]:
            line_clean = line.strip()
            if (len(line_clean) > 2 and len(line_clean) < 50 and
                not any(word in line_clean.lower() for word in ['resume', 'cv', 'curriculum', 'vitae', 'email', 'phone', 'linkedin']) and
                not re.search(r'\d', line_clean) and
                not re.search(r'[@#]', line_clean)):
                info["name"] = line_clean
                break

        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, resume_text)
        if emails:
            info["email"] = emails[0]

        # Extract phone number
        phone_pattern = r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}'
        phones = re.findall(phone_pattern, resume_text)
        if phones:
            phone_str = phones[0]
            if isinstance(phone_str, tuple):
                phone_str = ''.join(phone_str)
            info["phone"] = phone_str

        # Extract location
        location_keywords = ["location", "address", "city", "state", "country", "based in", "lives in"]
        lines = resume_text.lower().split('\n')
        for line in lines:
            for keyword in location_keywords:
                if keyword in line:
                    info["location"] = line.strip().replace(keyword, '').replace(':', '').strip()
                    break

        # Extract LinkedIn
        linkedin_pattern = r'(linkedin\.com/in/[\w-]+|linkedin\.com/company/[\w-]+)'
        linkedin_matches = re.findall(linkedin_pattern, resume_text, re.IGNORECASE)
        if linkedin_matches:
            info["linkedin"] = f"https://{linkedin_matches[0].lower()}"

        # Extract GitHub
        github_pattern = r'(github\.com/[\w-]+)'
        github_matches = re.findall(github_pattern, resume_text, re.IGNORECASE)
        if github_matches:
            info["github"] = f"https://{github_matches[0].lower()}"

        return info

    @staticmethod
    def extract_experience(resume_text):
        """Extract experience information"""
        experience = []
        keywords = ["experience", "work history", "employment", "professional experience", "work experience"]

        lines = resume_text.split('\n')
        in_experience_section = False
        section_count = 0

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Check if entering experience section
            for keyword in keywords:
                if keyword in line_lower and len(line.strip()) < 50 and section_count < 2:
                    in_experience_section = True
                    section_count += 1
                    continue

            if in_experience_section and line.strip():
                # Skip if this line starts another section
                if any(section in line_lower for section in ["education", "skills", "projects", "certifications"]):
                    break

                if len(line.strip()) > 10 and not line.strip().startswith(('•', '-', '*', '–')):
                    # Add bullet point indicator
                    experience.append(f"• {line.strip()}")
                elif len(line.strip()) > 10:
                    experience.append(line.strip())

                if len(experience) >= 8:
                    break

        return experience if experience else [
            "• [Job Title], [Company Name] | [City, State] | [Start Date] - [End Date]",
            "• [Responsibility/Achievement 1 with metrics]",
            "• [Responsibility/Achievement 2 with metrics]",
            "• [Responsibility/Achievement 3 with metrics]"
        ]

    @staticmethod
    def extract_education(resume_text):
        """Extract education information"""
        education = []
        keywords = ["education", "degree", "university", "college", "school", "bachelor", "master", "phd", "graduation"]

        lines = resume_text.split('\n')
        in_education_section = False
        section_count = 0

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Check if entering education section
            for keyword in keywords:
                if keyword in line_lower and len(line.strip()) < 50 and section_count < 2:
                    in_education_section = True
                    section_count += 1
                    continue

            if in_education_section and line.strip():
                # Skip if this line starts another section
                if any(section in line_lower for section in ["experience", "skills", "projects", "work"]):
                    break

                if len(line.strip()) > 10:
                    if not line.strip().startswith(('•', '-', '*', '–')):
                        education.append(f"• {line.strip()}")
                    else:
                        education.append(line.strip())

                if len(education) >= 4:
                    break

        return education if education else [
            "• [Degree Name], [Major]",
            "• [University Name], [City, State]",
            "• [Graduation Date] | GPA: [GPA]"
        ]

    @staticmethod
    def extract_projects(resume_text):
        """Extract project information"""
        projects = []
        keywords = ["projects", "personal projects", "academic projects", "project experience"]

        lines = resume_text.split('\n')
        in_project_section = False

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Check if entering project section
            for keyword in keywords:
                if keyword in line_lower and len(line.strip()) < 50:
                    in_project_section = True
                    continue

            if in_project_section and line.strip():
                if any(section in line_lower for section in ["education", "skills", "experience", "certifications"]):
                    break

                if len(line.strip()) > 15:
                    projects.append(line.strip())

                if len(projects) >= 6:
                    break

        return projects if projects else [
            "[Project Name]",
            "• [Project description and your contributions]",
            "• Technologies: [Tech Stack]",
            "• [GitHub/Deployment Link if available]"
        ]

    @staticmethod
    def extract_certifications(resume_text):
        """Extract certification information"""
        certifications = []
        keywords = ["certifications", "certificate", "licenses", "courses", "training"]

        lines = resume_text.split('\n')
        in_cert_section = False

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Check if entering certification section
            for keyword in keywords:
                if keyword in line_lower and len(line.strip()) < 50:
                    in_cert_section = True
                    continue

            if in_cert_section and line.strip():
                if any(section in line_lower for section in ["education", "skills", "experience", "projects"]):
                    break

                if len(line.strip()) > 10 and not line.strip().startswith(('•', '-', '*', '–')):
                    certifications.append(f"• {line.strip()}")
                elif len(line.strip()) > 10:
                    certifications.append(line.strip())

                if len(certifications) >= 4:
                    break

        return certifications if certifications else [
            "• [Certification Name], [Issuing Organization]",
            "• [Certification Name], [Issuing Organization]"
        ]

    @staticmethod
    def extract_skills_list(resume_text, skills_dict):
        """Extract categorized skills"""
        text_lower = resume_text.lower()
        extracted_skills = {}

        for category, skills in skills_dict.items():
            found_skills = []
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill.title())
            if found_skills:
                extracted_skills[category] = found_skills

        # Add uncategorized skills
        uncategorized = []
        skill_words = re.findall(r'\b[a-z]{4,15}\b', text_lower)
        common_words = set(['with', 'from', 'that', 'this', 'have', 'were', 'been', 'they', 'what'])

        for word in skill_words:
            if (word not in common_words and
                len(word) > 3 and
                not any(word in skill_list for skill_list in extracted_skills.values())):
                uncategorized.append(word.title())

        if uncategorized:
            extracted_skills["Other Skills"] = uncategorized[:10]

        return extracted_skills

    @staticmethod
    def create_resume_pdf(template_type, personal_info, experience, education, skills, projects, certifications,
                          category, profile_image=None):
        """Create a professional PDF resume"""

        # Create buffer for PDF
        buffer = io.BytesIO()

        # Create document with margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch*0.75,
            leftMargin=inch*0.75,
            topMargin=inch*0.75,
            bottomMargin=inch*0.75
        )

        # Get styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e3a8a'),
            spaceBefore=12,
            spaceAfter=6,
            borderWidth=1,
            borderColor=colors.HexColor('#d1d5db'),
            borderPadding=5,
            borderRadius=4,
            backgroundColor=colors.HexColor('#f3f4f6')
        )

        normal_style = ParagraphStyle(
            'NormalStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14
        )

        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            bulletIndent=10,
            spaceBefore=3,
            spaceAfter=3
        )

        # Story elements
        story = []

        # Header with optional profile image
        if profile_image and template_type != "ATS Optimized":
            try:
                # Create a table for header with image
                header_data = []
                if isinstance(profile_image, str) and os.path.exists(profile_image):
                    img = Image(profile_image, width=1.2*inch, height=1.2*inch)
                    header_data.append([[img], [""]])
                elif isinstance(profile_image, io.BytesIO):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(profile_image.getvalue())
                        tmp_path = tmp.name

                    img = Image(tmp_path, width=1.2*inch, height=1.2*inch)
                    header_data.append([[img], [""]])
                    os.unlink(tmp_path)
            except:
                pass

        # Add name
        story.append(Paragraph(personal_info['name'].upper(), title_style))

        # Add contact info
        contact_info = f"""
        <b>Email:</b> {personal_info['email']} | 
        <b>Phone:</b> {personal_info['phone']} | 
        <b>Location:</b> {personal_info['location']}
        """
        if personal_info['linkedin'] != '[LinkedIn URL]':
            contact_info += f" | <b>LinkedIn:</b> {personal_info['linkedin']}"
        if personal_info['github'] != '[GitHub URL]':
            contact_info += f" | <b>GitHub:</b> {personal_info['github']}"

        story.append(Paragraph(contact_info, normal_style))
        story.append(Spacer(1, 20))

        # Professional Summary
        story.append(Paragraph("<b>PROFESSIONAL SUMMARY</b>", heading_style))
        summary_text = f"""
        Results-driven {category} professional with expertise in {', '.join(list(skills.values())[0][:3] if skills else ['technical', 'analytical'])}. 
        Seeking to leverage skills and experience at innovative organizations.
        """
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 12))

        # Skills Section
        story.append(Paragraph("<b>TECHNICAL SKILLS</b>", heading_style))

        # Create skills table
        skill_items = []
        for category_name, skill_list in skills.items():
            if skill_list:
                skill_text = f"<b>{category_name}:</b> {', '.join(skill_list[:8])}"
                skill_items.append(Paragraph(skill_text, normal_style))

        # Layout skills in columns
        if skill_items:
            skill_table_data = []
            for i in range(0, len(skill_items), 2):
                row = []
                row.append(skill_items[i])
                if i + 1 < len(skill_items):
                    row.append(skill_items[i + 1])
                else:
                    row.append(Paragraph("", normal_style))
                skill_table_data.append(row)

            skill_table = Table(skill_table_data, colWidths=[doc.width/2, doc.width/2])
            skill_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(skill_table)

        story.append(Spacer(1, 12))

        # Work Experience
        story.append(Paragraph("<b>WORK EXPERIENCE</b>", heading_style))
        for exp in experience[:4]:
            if exp.strip():
                story.append(Paragraph(f"• {exp}", bullet_style))
        story.append(Spacer(1, 12))

        # Education
        story.append(Paragraph("<b>EDUCATION</b>", heading_style))
        for edu in education[:3]:
            if edu.strip():
                story.append(Paragraph(f"• {edu}", bullet_style))
        story.append(Spacer(1, 12))

        # Projects (if any)
        if projects and len(projects) > 0 and projects[0] != '[Project Name]':
            story.append(Paragraph("<b>PROJECTS</b>", heading_style))
            for proj in projects[:3]:
                if proj.strip():
                    story.append(Paragraph(f"• {proj}", bullet_style))
            story.append(Spacer(1, 12))

        # Certifications (if any)
        if certifications and len(certifications) > 0 and certifications[0] != '• [Certification Name], [Issuing Organization]':
            story.append(Paragraph("<b>CERTIFICATIONS</b>", heading_style))
            for cert in certifications[:3]:
                if cert.strip():
                    story.append(Paragraph(f"• {cert}", bullet_style))

        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Generated by AI Resume Analyzer | {datetime.now().strftime('%B %d, %Y')}"
        story.append(Paragraph(footer_text, ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )))

        # Build PDF
        doc.build(story)

        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    @staticmethod
    def generate_visual_resume(resume_text, category, skills):
        """Generate a visually appealing resume with charts"""
        # Create a matplotlib figure
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Skill distribution pie chart
        if skills:
            labels = list(skills.keys())[:5]
            sizes = [len(v) for v in list(skills.values())[:5]]
            ax[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax[0].set_title('Top Skill Categories')

        # Skill count bar chart
        if skills:
            categories = list(skills.keys())[:6]
            counts = [len(v) for v in list(skills.values())[:6]]
            ax[1].bar(categories, counts, color='skyblue')
            ax[1].set_title('Skills by Category')
            ax[1].tick_params(axis='x', rotation=45)

        # Resume strength gauge
        ax[2].axis('off')
        ax[2].text(0.5, 0.7, f'Resume Score', ha='center', fontsize=14)
        ax[2].text(0.5, 0.5, f'{np.random.randint(70, 95)}/100', ha='center', fontsize=24, fontweight='bold')
        ax[2].text(0.5, 0.3, f'Category: {category}', ha='center', fontsize=12)

        plt.tight_layout()

        # Save figure to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        return img_buffer

    @staticmethod
    def create_html_resume(template_type, personal_info, experience, education, skills, category):
        """Create an HTML resume"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{personal_info['name']} - Resume</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #2563eb;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .name {{
                    font-size: 32px;
                    color: #1e40af;
                    margin-bottom: 10px;
                }}
                .contact-info {{
                    display: flex;
                    justify-content: center;
                    flex-wrap: wrap;
                    gap: 15px;
                    font-size: 14px;
                }}
                .section {{
                    margin-bottom: 25px;
                }}
                .section-title {{
                    font-size: 18px;
                    color: #1e3a8a;
                    border-bottom: 2px solid #e5e7eb;
                    padding-bottom: 5px;
                    margin-bottom: 15px;
                }}
                .skills-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 10px;
                }}
                .skill-category {{
                    background: #f3f4f6;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #e5e7eb;
                    font-size: 12px;
                    color: #6b7280;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 class="name">{personal_info['name']}</h1>
                <div class="contact-info">
                    <span>📧 {personal_info['email']}</span>
                    <span>📱 {personal_info['phone']}</span>
                    <span>📍 {personal_info['location']}</span>
                    {f'<span><a href="{personal_info["linkedin"]}">🔗 LinkedIn</a></span>' if personal_info['linkedin'] != '[LinkedIn URL]' else ''}
                    {f'<span><a href="{personal_info["github"]}">💻 GitHub</a></span>' if personal_info['github'] != '[GitHub URL]' else ''}
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Professional Summary</h2>
                <p>Results-driven {category} with expertise in key technologies. Seeking to leverage skills in a challenging role.</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">Skills</h2>
                <div class="skills-grid">
        """

        for category_name, skill_list in skills.items():
            html_template += f"""
                    <div class="skill-category">
                        <strong>{category_name}:</strong><br>
                        {', '.join(skill_list[:6])}
                    </div>
            """

        html_template += """
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Experience</h2>
                <ul>
        """

        for exp in experience[:3]:
            html_template += f"""
                    <li>{exp}</li>
            """

        html_template += """
                </ul>
            </div>
            
            <div class="section">
                <h2 class="section-title">Education</h2>
                <ul>
        """

        for edu in education[:2]:
            html_template += f"""
                    <li>{edu}</li>
            """

        html_template += f"""
                </ul>
            </div>
            
            <div class="footer">
                Generated by AI Resume Analyzer | {datetime.now().strftime('%B %d, %Y')}
            </div>
        </body>
        </html>
        """

        return html_template

# Initialize enhanced template engine
template_engine = EnhancedResumeTemplateEngine()

# =========================
# CSS with Animations and Smart Navigation
# =========================
def load_css(mode, accent):
    bg = "#0f172a" if mode == "Dark" else "#f5f7fb"
    card = "#111827" if mode == "Dark" else "#ffffff"
    text = "#e5e7eb" if mode == "Dark" else "#111827"
    soft = "#1f2933" if mode == "Dark" else "#e5e7eb"

    st.markdown(f"""
    <style>
    /* App Background with Gradient Animation */
    .stApp {{
        background: linear-gradient(135deg, {bg}, #020617);
        color: {text};
        font-family: 'Segoe UI', sans-serif;
        animation: gradientShift 20s ease infinite;
        background-size: 400% 400%;
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Login Container with Animation */
    .login-container {{
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: {card};
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        animation: slideIn 0.8s ease-out;
    }}
    
    @keyframes slideIn {{
        from {{ 
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{ 
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .login-title {{
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 30px;
        background: linear-gradient(90deg, {accent}, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titlePulse 3s infinite;
    }}
    
    @keyframes titlePulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}

    /* Cards with Hover Animation */
    .card {{
        background: {card};
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid {soft};
    }}

    .card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.35);
        border-color: {accent};
    }}

    /* KPI Cards with Animation */
    .kpi {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, {accent}, #2563eb);
        color: white;
        padding: 20px;
        border-radius: 18px;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 10px 25px rgba(37,99,235,0.45);
        animation: kpiGlow 2s infinite alternate;
        transition: all 0.3s ease;
    }}
    
    @keyframes kpiGlow {{
        from {{ box-shadow: 0 10px 25px rgba(37,99,235,0.45); }}
        to {{ box-shadow: 0 15px 35px rgba(37,99,235,0.65); }}
    }}

    .kpi:hover {{
        transform: scale(1.05);
    }}

    .kpi span {{
        font-size: 14px;
        opacity: 0.85;
    }}

    /* Buttons with Enhanced Animations */
    .stButton > button {{
        background: linear-gradient(135deg, {accent}, #2563eb);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 18px rgba(37,99,235,0.35);
        position: relative;
        overflow: hidden;
    }}

    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 28px rgba(37,99,235,0.55);
        filter: brightness(1.15);
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px) scale(0.98);
    }}
    
    .stButton > button::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: rgba(255, 255, 255, 0.5);
        opacity: 0;
        border-radius: 100%;
        transform: scale(1, 1) translate(-50%);
        transform-origin: 50% 50%;
    }}
    
    .stButton > button:focus:not(:active)::after {{
        animation: ripple 1s ease-out;
    }}
    
    @keyframes ripple {{
        0% {{
            transform: scale(0, 0);
            opacity: 0.5;
        }}
        20% {{
            transform: scale(25, 25);
            opacity: 0.3;
        }}
        100% {{
            opacity: 0;
            transform: scale(40, 40);
        }}
    }}

    /* Template Cards */
    .template-card {{
        border: 2px solid {accent};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .template-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: 0.5s;
    }}
    
    .template-card:hover::before {{
        left: 100%;
    }}

    .template-card:hover {{
        background: rgba(37, 99, 235, 0.1);
        transform: translateY(-4px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }}

    /* Progress bar with Animation */
    div[role="progressbar"] > div {{
        background: linear-gradient(90deg, {accent}, #22c55e);
        border-radius: 10px;
        animation: progressFill 2s ease-out;
    }}
    
    @keyframes progressFill {{
        from {{ width: 0%; }}
        to {{ width: var(--progress-width); }}
    }}

    /* Enhanced Tabs with Animation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background: {soft};
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 20px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .stTabs [data-baseweb="tab"]::before {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: {accent};
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover::before {{
        transform: scaleX(0.3);
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {accent} !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37,99,235,0.3);
    }}
    
    .stTabs [aria-selected="true"]::before {{
        transform: scaleX(1);
    }}
    
    /* Tab Content Animation */
    .tab-content {{
        animation: fadeIn 0.5s ease-out;
    }}
    
    @keyframes fadeIn {{
        from {{ 
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{ 
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* Auto-fill Button Special */
    .auto-fill-btn {{
        background: linear-gradient(135deg, #10b981, #059669) !important;
        animation: pulseGlow 2s infinite;
    }}
    
    @keyframes pulseGlow {{
        0%, 100% {{ box-shadow: 0 6px 18px rgba(16, 185, 129, 0.35); }}
        50% {{ box-shadow: 0 6px 18px rgba(16, 185, 129, 0.65); }}
    }}
    
    .auto-fill-btn:hover {{
        background: linear-gradient(135deg, #059669, #047857) !important;
        animation: none;
    }}

    /* Download Button Special */
    .download-btn {{
        background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
    }}
    
    .download-btn:hover {{
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    }}

    /* Smart Navigation Bar */
    .smart-nav {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: {card};
        padding: 15px 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid {soft};
        position: sticky;
        top: 10px;
        z-index: 100;
    }}
    
    .nav-item {{
        padding: 10px 20px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        flex: 1;
        margin: 0 5px;
        background: {soft};
        color: {text};
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    
    .nav-item:hover {{
        background: rgba(37, 99, 235, 0.1);
        transform: translateY(-2px);
        border: 1px solid {accent};
    }}
    
    .nav-item.active {{
        background: linear-gradient(135deg, {accent}, #2563eb);
        color: white;
        box-shadow: 0 5px 15px rgba(37,99,235,0.3);
        transform: translateY(-2px);
    }}
    
    .nav-icon {{
        font-size: 18px;
    }}

    /* Celebration Animations */
    .celebration {{
        animation: celebrate 2s ease-out;
    }}
    
    @keyframes celebrate {{
        0% {{ transform: scale(0.8); opacity: 0; }}
        50% {{ transform: scale(1.1); }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}

    /* Floating Elements */
    .floating {{
        animation: float 6s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}

    /* Success Message Animation */
    .success-message {{
        animation: successPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }}
    
    @keyframes successPop {{
        0% {{ transform: scale(0.5); opacity: 0; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}

    /* Loading Animation */
    .loading-spinner {{
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: {accent};
        animation: spin 1s ease-in-out infinite;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# =========================
# Resume Templates Dictionary
# =========================
RESUME_TEMPLATES = {
    "Modern Professional": """[YOUR NAME]
[City, State] | [Phone] | [Email] | [LinkedIn URL] | [Portfolio URL]

PROFESSIONAL SUMMARY
Results-driven [Job Title] with [X] years of experience in [Industry/Field]. Proven track record of [Achievement 1] and [Achievement 2]. Seeking to leverage skills in [Skill 1], [Skill 2], and [Skill 3] at [Company Name].

TECHNICAL SKILLS
• Programming: [Languages]
• Frameworks: [Framework 1, Framework 2]
• Tools: [Tool 1, Tool 2, Tool 3]
• Databases: [DB1, DB2]
• Other: [Skill 1, Skill 2]

WORK EXPERIENCE
[Job Title], [Company Name] | [City, State] | [Start Date] - [Present/End Date]
• [Achievement/Bullet Point 1 with metrics]
• [Achievement/Bullet Point 2 with metrics]
• [Achievement/Bullet Point 3 with metrics]

[Job Title], [Company Name] | [City, State] | [Start Date] - [End Date]
• [Achievement/Bullet Point 1]
• [Achievement/Bullet Point 2]
• [Achievement/Bullet Point 3]

EDUCATION
[Degree Name], [Major]
[University Name], [City, State]
[Graduation Date]
GPA: [GPA] | Relevant Coursework: [Course 1, Course 2, Course 3]

PROJECTS
[Project Name]
• [Project Description]
• Technologies: [Tech Stack]
• [Link to GitHub/Deployment]

[Project Name]
• [Project Description]
• Technologies: [Tech Stack]

CERTIFICATIONS
• [Certification 1] | [Issuing Organization] | [Year]
• [Certification 2] | [Issuing Organization] | [Year]

ADDITIONAL INFORMATION
• Languages: [Language 1] (Proficient), [Language 2] (Intermediate)
• Interests: [Interest 1], [Interest 2], [Interest 3]""",

    "Clean Minimalist": """[YOUR NAME]
[Email] | [Phone] | [Location] | [LinkedIn]

SKILLS
Technical: [Skill 1, Skill 2, Skill 3, Skill 4]
Tools: [Tool 1, Tool 2, Tool 3]
Soft Skills: [Communication, Problem Solving, Leadership]
Databases: [Database 1, Database 2]

EXPERIENCE
[Company] | [Role]
[Dates]
- [Responsibility/Achievement with metrics]
- [Responsibility/Achievement with metrics]
- [Responsibility/Achievement with metrics]

[Company] | [Role]
[Dates]
- [Responsibility/Achievement]
- [Responsibility/Achievement]

EDUCATION
[Degree], [University]
[Year]
[Relevant Details/Awards]

PROJECTS
[Project Title]
[Brief Description]
[Technologies Used | GitHub Link]

[Project Title]
[Brief Description]
[Technologies Used]

CERTIFICATIONS
[Certification Name] | [Issuing Organization] | [Year]
[Certification Name] | [Issuing Organization] | [Year]

INTERESTS
[Interest 1] • [Interest 2] • [Interest 3]""",

    "ATS Optimized": """[FIRST NAME] [LAST NAME]
[City, State ZIP Code] | [Phone Number] | [Email Address] | [LinkedIn Profile URL]

PROFESSIONAL PROFILE
Dedicated and performance-driven [Job Title] professional with extensive experience in [Industry]. Recognized for [Key Strength 1] and [Key Strength 2]. Seeking to contribute to [Company Name]'s success through [Specific Skill].

CORE COMPETENCIES
• [Keyword 1 from Job Description]
• [Keyword 2 from Job Description]
• [Keyword 3 from Job Description]
• [Keyword 4 from Job Description]
• [Keyword 5 from Job Description]

PROFESSIONAL EXPERIENCE
[Company Name], [Location] — [Job Title]
[Month Year] to Present
• [Action Verb] [Task] resulting in [Quantifiable Result]
• [Action Verb] [Task] resulting in [Quantifiable Result]
• [Action Verb] [Task] resulting in [Quantifiable Result]

[Company Name], [Location] — [Job Title]
[Month Year] to [Month Year]
• [Action Verb] [Task] resulting in [Quantifiable Result]
• [Action Verb] [Task] resulting in [Quantifiable Result]

EDUCATION
[Degree Type] in [Major]
[University Name], [Location] — [Year]
[Relevant Coursework: Course 1, Course 2, Course 3]

TECHNICAL SKILLS
[Category 1]: [Skill 1], [Skill 2], [Skill 3]
[Category 2]: [Skill 1], [Skill 2], [Skill 3]
[Category 3]: [Skill 1], [Skill 2]

PROJECTS
[Project Name]
• [Description highlighting technical skills and achievements]
• Technologies: [Technology List]

CERTIFICATIONS & TRAINING
• [Certification Name], [Issuing Organization] ([Year])
• [Certification Name], [Issuing Organization] ([Year])

ADDITIONAL INFORMATION
• Technical Writing | Public Speaking | Team Leadership | Agile Methodologies""",

    "Academic/Entry Level": """[YOUR NAME]
[Email] | [Phone] | [LinkedIn] | [GitHub] | [Portfolio]

EDUCATION
[Degree], [Major]
[University Name], [City, State]
[Expected Graduation: Month Year]
GPA: [GPA] | Relevant Coursework: [Course 1], [Course 2], [Course 3], [Course 4]

ACADEMIC PROJECTS
[Project Title]
[University/Personal Project] | [Month Year] - [Month Year]
• [Detailed description of project goals and your contributions]
• Technologies: [List of technologies used]
• [Link to GitHub/Project URL]

[Project Title]
[Hackathon/Course Project] | [Month Year]
• [Brief description and your role]
• [Key technologies or methods used]
• [Outcomes/Results achieved]

SKILLS
Technical: [Programming Languages], [Frameworks/Libraries], [Tools/Software]
Laboratory: [Lab Techniques], [Equipment], [Methodologies]
Languages: [Language 1] (Proficient), [Language 2] (Intermediate)
Soft Skills: [Communication], [Teamwork], [Problem Solving]

EXPERIENCE
[Position Title], [Organization]
[City, State] | [Start Date] - [End Date]
• [Responsibility or achievement using action verbs]
• [Quantify results if possible]

[Position Title], [Organization]
[City, State] | [Start Date] - [End Date]
• [Responsibility or achievement]

LEADERSHIP & ACTIVITIES
[Position], [Club/Organization]
[University Name] | [Dates]
• [Description of responsibilities and achievements]

[Position], [Volunteer Organization]
[Location] | [Dates]
• [Description of contributions]

CERTIFICATIONS & AWARDS
• [Certification Name] | [Issuing Body] | [Year]
• [Award Name] | [Organization] | [Year]

RESEARCH & PUBLICATIONS
[Research Title] (if applicable)
• [Brief description]
• [Conference/Journal Name] | [Year]"""
}

# =========================
# Utilities
# =========================
def cleanResume(text):
    text = re.sub(r"http\S+|@\S+|#\S+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\w\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

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
    if ext == "pdf": return extract_text_from_pdf(uploaded_file)
    if ext == "docx": return extract_text_from_docx(uploaded_file)
    if ext == "txt": return extract_text_from_txt(uploaded_file)
    if ext in ["htm", "html"]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(uploaded_file.read(), "html.parser")
        return soup.get_text()
    raise ValueError("Unsupported file format")

# =========================
# ML Functions - FIXED VERSION
# =========================
def category_confidence(resume_text):
    text = resume_text.lower()

    role_scores = {
        role: sum(1 for kw in kws if kw in text)
        for role, kws in ROLE_SKILLS.items()
    }

    best_role = max(role_scores, key=role_scores.get)

    if role_scores[best_role] > 0:
        confidence = min(role_scores[best_role] * 15, 95)
        return best_role, confidence

    try:
        # Fixed line: Changed filename(resume_text) to cleanResume(resume_text)
        vec = tfidf.transform([cleanResume(resume_text)]).toarray()
        scores = svc_model.decision_function(vec)[0]
        idx = np.argmax(scores)

        confidence = round(
            (scores[idx] - scores.min()) / (scores.max() - scores.min() + 1e-9) * 100,
            2
        )

        return le.inverse_transform([idx])[0], confidence
    except Exception as e:
        # Fallback if ML model fails
        return best_role, 70.0

def predict_category(resume_text):
    """Predict job category with robust error handling"""
    try:
        # First check if we have real models loaded
        if hasattr(tfidf, 'transform') and hasattr(svc_model, 'predict'):
            # Clean the resume text
            cleaned_text = cleanResume(resume_text)
            
            # Transform using TF-IDF - FIXED: Changed filename() to cleanResume()
            vec = tfidf.transform([cleaned_text]).toarray()
            
            # Make prediction
            pred = svc_model.predict(vec)
            
            # Decode category
            category = le.inverse_transform(pred)[0]
            return category
        else:
            # Use fallback method if models are dummy
            return fallback_category_detection(resume_text)
    except AttributeError as e:
        st.warning(f"ML model error: {str(e)}. Using fallback detection.")
        return fallback_category_detection(resume_text)
    except Exception as e:
        st.warning(f"Error in category prediction: {str(e)}")
        return fallback_category_detection(resume_text)

def fallback_category_detection(resume_text):
    """Fallback method if ML model fails"""
    text = resume_text.lower()
    
    # Check for role keywords
    role_keywords = {
        "Data Scientist": ["data", "machine learning", "python", "analysis", "statistics"],
        "Software Developer": ["developer", "programming", "code", "software", "java", "python"],
        "Web Developer": ["web", "frontend", "backend", "html", "css", "javascript"],
        "DevOps Engineer": ["devops", "docker", "kubernetes", "aws", "cloud", "ci/cd"],
        "Data Analyst": ["analyst", "excel", "sql", "reporting", "tableau", "power bi"],
        "Java Developer": ["java", "spring", "hibernate", "j2ee"],
        "Python Developer": ["python", "django", "flask", "fastapi"],
        "Frontend Developer": ["react", "angular", "vue", "javascript", "typescript"],
        "Backend Developer": ["node.js", "express", "api", "microservices"],
        "Full Stack Developer": ["full stack", "mern", "mean", "react", "node"],
        "Machine Learning Engineer": ["machine learning", "deep learning", "tensorflow", "pytorch"],
        "Cloud Engineer": ["aws", "azure", "gcp", "cloud", "terraform"],
        "Project Manager": ["project", "management", "agile", "scrum", "jira"],
        "System Administrator": ["system admin", "linux", "windows server", "networking"]
    }
    
    scores = {}
    for role, keywords in role_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[role] = score
    
    if scores:
        best_role = max(scores, key=scores.get)
        return best_role
    else:
        # Default category
        return "Software Developer"

def extract_skills(text):
    text = text.lower()
    return {k: [s for s in v if s in text] for k, v in SKILLS_DB.items() if any(s in text for s in v)}

def resume_strength_score(text, skills):
    score = min(sum(len(v) for v in skills.values()) * 4, 40)
    if "project" in text: score += 20
    if "intern" in text or "experience" in text: score += 20
    if "certificate" in text: score += 20
    if "education" in text: score += 10
    if "certification" in text: score += 10
    if "leadership" in text or "volunteer" in text: score += 10
    if "achievement" in text or "award" in text: score += 10
    if "github" in text or "portfolio" in text: score += 10
    if "communication" in text or "teamwork" in text: score += 10
    if "problem solving" in text or "critical thinking" in text: score += 10
    if "adaptability" in text or "flexibility" in text: score += 10
    if "time management" in text or "organization" in text: score += 10
    if "leadership" in text or "initiative" in text: score += 10
    if "certified" in text: score += 10
    if "published" in text or "research" in text: score += 10
    if "awards" in text or "recognition" in text: score += 10
    if "extracurricular" in text or "clubs" in text: score += 10
    return min(score, 100)

# =========================
# Charts
# =========================
def plot_skill_pie(skills):
    if not skills:
        return None
    labels = list(skills.keys())
    sizes = [len(v) for v in skills.values()]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        textprops={"fontsize": 8}
    )
    centre_circle = Circle((0, 0), 0.70, fc="white")
    ax.add_artist(centre_circle)
    ax.axis("equal")
    return fig

def radar_chart(skills):
    cats = list(SKILLS_DB.keys())
    values = [len(skills.get(c, [])) for c in cats]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(cats)+1)
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_thetagrids(angles[:-1]*180/np.pi, cats, fontsize=8)
    return fig

# =========================
# ATS Functions
# =========================
def ats_score(resume, jd):
    try:
        r = tfidf.transform([cleanResume(resume)])
        j = tfidf.transform([cleanResume(jd)])
        return round(cosine_similarity(r, j)[0][0] * 100, 2)
    except:
        # Simple keyword matching fallback
        resume_lower = resume.lower()
        jd_lower = jd.lower()
        
        resume_words = set(re.findall(r'\b\w+\b', resume_lower))
        jd_words = set(re.findall(r'\b\w+\b', jd_lower))
        
        common_words = resume_words.intersection(jd_words)
        if len(jd_words) > 0:
            score = len(common_words) / len(jd_words) * 100
            return round(min(score, 100), 2)
        return 50.0

def resume_suggestions(score, skills):
    tips = []
    if score < 60: tips.append("Add more projects and technical depth.")
    if "Cloud" not in skills: tips.append("Add basic cloud exposure.")
    if "Visualization" not in skills: tips.append("Mention Power BI / Tableau.")
    if not tips: tips.append("Resume looks strong. Tailor it to job roles.")
    return tips

# =========================
# Authentication Pages with Celebration
# =========================
def login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">🔐 AI Resume Analyzer</h1>', unsafe_allow_html=True)

    # Show balloons on login page load
    if 'login_celebration' not in st.session_state:
        st.session_state.login_celebration = True
        st.balloons()

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                success, message = auth.login(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.show_celebration = True
                    st.session_state.current_tab = "analysis"  # Default tab
                    st.success(message)
                    # Show celebration immediately
                    st.balloons()
                    time.sleep(0.5)  # Short delay for animation
                    st.rerun()
                else:
                    st.error(message)

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Register")

            if submit_register:
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = auth.register(new_username, new_password, new_email)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = new_username
                        st.session_state.show_celebration = True
                        st.session_state.current_tab = "analysis"
                        st.success(message)
                        st.balloons()
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(message)

    st.markdown('</div>', unsafe_allow_html=True)

    # Demo credentials
    with st.expander("Demo Credentials"):
        st.write("**Username:** demo")
        st.write("**Password:** demo123")

# =========================
# Enhanced Template Selector with PDF and Image Generation
# =========================
def enhanced_template_selector(tab_name="analysis", resume_text="", category=""):
    """Create enhanced template selector with PDF, HTML, and image generation"""

    st.subheader("📋 Resume Template Generator")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        template_name = st.selectbox(
            "Select Template Style",
            list(RESUME_TEMPLATES.keys()),
            key=f"template_select_{tab_name}"
        )

    with col2:
        output_format = st.selectbox(
            "Output Format",
            ["PDF", "HTML", "Image", "Text"],
            key=f"output_format_{tab_name}"
        )

    with col3:
        profile_image = None
        use_profile_image = st.checkbox("Add Profile Image", key=f"use_image_{tab_name}")
        if use_profile_image:
            profile_image = st.file_uploader(
                "Upload Profile Image",
                type=["jpg", "jpeg", "png"],
                key=f"profile_image_{tab_name}"
            )

    # Auto-fill section
    if resume_text and category:
        st.markdown("---")
        st.subheader("✨ Auto-Fill Options")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Extract Information",
                        key=f"extract_btn_{tab_name}",
                        use_container_width=True):

                # Extract information
                personal_info = template_engine.extract_personal_info(resume_text)
                experience = template_engine.extract_experience(resume_text)
                education = template_engine.extract_education(resume_text)
                projects = template_engine.extract_projects(resume_text)
                certifications = template_engine.extract_certifications(resume_text)
                skills = template_engine.extract_skills_list(resume_text, SKILLS_DB)

                # Store in session state
                st.session_state.auto_filled_data = {
                    "personal_info": personal_info,
                    "experience": experience,
                    "education": education,
                    "projects": projects,
                    "certifications": certifications,
                    "skills": skills,
                    "category": category
                }

                st.success("Information extracted successfully!")

                # Show extracted info
                with st.expander("View Extracted Information", expanded=True):
                    st.write("**Personal Info:**")
                    st.json(personal_info)
                    st.write(f"**Category:** {category}")
                    st.write(f"**Experience Points:** {len(experience)}")
                    st.write(f"**Education Points:** {len(education)}")
                    st.write(f"**Skills Categories:** {len(skills)}")

        with col2:
            if st.session_state.get('auto_filled_data'):
                if st.button("Generate Resume",
                           key=f"generate_btn_{tab_name}",
                           type="primary",
                           use_container_width=True):

                    data = st.session_state.auto_filled_data

                    # Handle profile image
                    profile_img_data = None
                    if profile_image:
                        profile_img_data = io.BytesIO(profile_image.getvalue())

                    # Generate based on output format
                    if output_format == "PDF":
                        # Generate PDF
                        pdf_data = template_engine.create_resume_pdf(
                            template_name,
                            data["personal_info"],
                            data["experience"],
                            data["education"],
                            data["skills"],
                            data["projects"],
                            data["certifications"],
                            data["category"],
                            profile_img_data
                        )

                        # Download button for PDF
                        st.download_button(
                            label="📥 Download PDF Resume",
                            data=pdf_data,
                            file_name=f"Resume_{data['personal_info']['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_{tab_name}"
                        )

                        # Show PDF preview (first page as image)
                        st.info("PDF generated successfully! Click download button above.")

                    elif output_format == "HTML":
                        # Generate HTML
                        html_content = template_engine.create_html_resume(
                            template_name,
                            data["personal_info"],
                            data["experience"],
                            data["education"],
                            data["skills"],
                            data["category"]
                        )

                        # Download button for HTML
                        st.download_button(
                            label="📥 Download HTML Resume",
                            data=html_content,
                            file_name=f"Resume_{data['personal_info']['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                            key=f"download_html_{tab_name}"
                        )

                        # Show HTML preview
                        with st.expander("HTML Preview", expanded=True):
                            st.components.v1.html(html_content, height=600, scrolling=True)

                    elif output_format == "Image":
                        # Generate visual resume image
                        img_buffer = template_engine.generate_visual_resume(
                            resume_text,
                            data["category"],
                            data["skills"]
                        )

                        # Download button for image
                        st.download_button(
                            label="📥 Download Resume Image",
                            data=img_buffer,
                            file_name=f"Resume_Visual_{datetime.now().strftime('%Y%m%d')}.png",
                            mime="image/png",
                            key=f"download_img_{tab_name}"
                        )

                        # Show image preview
                        st.image(img_buffer, caption="Resume Visualization")

                    elif output_format == "Text":
                        # Generate text resume
                        text_resume = f"""{data['personal_info']['name']}
{data['personal_info']['email']} | {data['personal_info']['phone']} | {data['personal_info']['location']}

PROFESSIONAL SUMMARY
Results-driven {data['category']} with expertise in key technologies.

SKILLS"""

                        for cat, skill_list in data['skills'].items():
                            text_resume += f"\n• {cat}: {', '.join(skill_list[:5])}"

                        text_resume += "\n\nEXPERIENCE"
                        for exp in data['experience'][:3]:
                            text_resume += f"\n{exp}"

                        text_resume += "\n\nEDUCATION"
                        for edu in data['education'][:2]:
                            text_resume += f"\n{edu}"

                        # Download button for text
                        st.download_button(
                            label="📥 Download Text Resume",
                            data=text_resume,
                            file_name=f"Resume_{data['personal_info']['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            key=f"download_txt_{tab_name}"
                        )

                        # Show text preview
                        st.text_area("Resume Preview", text_resume, height=400)

    # Template preview section
    st.markdown("---")
    st.subheader("📄 Template Preview")

    preview_col1, preview_col2 = st.columns([3, 1])

    with preview_col1:
        st.text_area(
            f"{template_name} Template",
            RESUME_TEMPLATES[template_name],
            height=400,
            key=f"template_preview_{tab_name}"
        )

    with preview_col2:
        st.markdown("### Quick Actions")

        # Direct template download
        if st.download_button(
            label="📥 Download Template",
            data=RESUME_TEMPLATES[template_name],
            file_name=f"{template_name.replace(' ', '_')}_Template.txt",
            mime="text/plain",
            key=f"direct_download_{tab_name}",
            use_container_width=True
        ):
            st.success("Template downloaded!")

        # Use template button
        if st.button(
            "✏️ Use This Template",
            key=f"use_template_{tab_name}",
            use_container_width=True
        ):
            st.session_state.selected_template = template_name
            st.session_state.template_text = RESUME_TEMPLATES[template_name]
            st.success(f"Selected '{template_name}' template!")
            st.balloons()

        # Clear auto-fill data
        if st.session_state.get('auto_filled_data'):
            if st.button(
                "🗑️ Clear Extracted Data",
                key=f"clear_data_{tab_name}",
                use_container_width=True
            ):
                del st.session_state.auto_filled_data
                st.success("Extracted data cleared!")
                st.rerun()

# =========================
# Render Analysis Tab
# =========================
def render_analysis_tab():
    """Render the Resume Analysis tab with enhanced templates"""
    st.subheader("📊 Resume Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf","docx","txt"],
            key="analysis_file_uploader"
        )
        resume_text = st.text_area(
            "Or paste resume text here:",
            height=200,
            key="analysis_resume_text_area",
            placeholder="Paste your resume text here or upload a file..."
        )

        if uploaded_file:
            resume_text = handle_file_upload(uploaded_file)
            # Store in session state for template auto-fill
            st.session_state.current_resume_text = resume_text

    with col2:
        if resume_text:
            # Store extracted info for template auto-fill
            category = predict_category(resume_text)
            st.session_state.current_category = category

            # Use enhanced template selector
            enhanced_template_selector("analysis", resume_text, category)
        else:
            # Show basic template selector without auto-fill
            st.subheader("📋 Resume Templates")
            template_name = st.selectbox(
                "Select Template",
                list(RESUME_TEMPLATES.keys()),
                key="analysis_template_select"
            )

            if st.button("Preview Template", key="analysis_preview_btn"):
                st.text_area("Template Preview",
                           RESUME_TEMPLATES[template_name],
                           height=300,
                           key="analysis_template_preview")

    if resume_text:
        # Analysis
        category = predict_category(resume_text)
        cat, conf = category_confidence(resume_text)

        # Display confidence bar with animation
        st.markdown("### 🎨 Category Confidence")
        st.markdown(f"""
        <div style="
            width: 100%;
            background: #e5e7eb;
            border-radius: 12px;
            overflow: hidden;
            height: 26px;
            margin-top: 10px;
            margin-bottom: 20px;
        ">
            <div style="
                height: 100%;
                width: {conf}%;
                background: linear-gradient(90deg, #22c55e, #38bdf8, #6366f1);
                animation: progressFill 1.5s ease-out;
            "></div>
        </div>
        <p><b>{conf}% Confidence</b></p>
        <style>
        @keyframes progressFill {{
            from {{ width: 0%; }}
            to {{ width: {conf}%; }}
        }}
        </style>
        """, unsafe_allow_html=True)

        skills = extract_skills(resume_text)
        score = resume_strength_score(resume_text, skills)

        # KPI Cards with animations
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="card celebration">', unsafe_allow_html=True)
            st.metric("📄 Category", category)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card celebration">', unsafe_allow_html=True)
            st.metric("🧠 Strength", f"{score}/100")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card celebration">', unsafe_allow_html=True)
            st.metric("🛠 Skills", sum(len(v) for v in skills.values()))
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="card celebration">', unsafe_allow_html=True)
            st.metric("📈 Confidence", f"{conf}%" if conf else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        # Show extraction summary
        if resume_text:
            personal_info = template_engine.extract_personal_info(resume_text)
            experience = template_engine.extract_experience(resume_text)
            education = template_engine.extract_education(resume_text)

            extraction_summary = f"""
            ### 📊 Extracted Information
            
            **Contact Information:**
            - Name: {personal_info['name']}
            - Email: {personal_info['email']}
            - Phone: {personal_info['phone']}
            - Location: {personal_info['location']}
            
            **Experience Points:** {len(experience)}
            **Education Details:** {len(education)}
            **Skills Categories:** {len(skills)}
            **Total Skills:** {sum(len(v) for v in skills.values())}
            
            *Ready for auto-fill! Click the "Extract Information" button above.*
            """

            with st.expander("📊 View Extracted Information", expanded=False):
                st.markdown(extraction_summary)

        st.divider()

        # Detailed Analysis Tabs
        anal_tab1, anal_tab2, anal_tab3, anal_tab4 = st.tabs(
            ["📄 Skills Analysis", "📊 Visualizations", "📈 ATS Matching", "🤖 Suggestions"])

        with anal_tab1:
            st.subheader("Extracted Skills by Category")
            if skills:
                for k, v in skills.items():
                    if v:
                        st.markdown(f"**🔹 {k}:** {' '.join(f'`{s}`' for s in v)}")
            else:
                st.info("No skills detected in the resume text.")

        with anal_tab2:
            col1, col2 = st.columns(2)
            with col1:
                if skills:
                    st.pyplot(plot_skill_pie(skills))
                else:
                    st.info("No skills detected for visualization")
            with col2:
                if skills:
                    st.pyplot(radar_chart(skills))
                else:
                    st.info("No skills detected for radar chart")

            # Generate visual resume image option
            st.markdown("---")
            st.subheader("🎨 Generate Visual Resume")

            if st.button("Create Visual Resume Image", key="generate_visual_resume"):
                img_buffer = template_engine.generate_visual_resume(resume_text, category, skills)
                st.image(img_buffer, caption="Your Visual Resume")

                # Download button for the image
                st.download_button(
                    label="📥 Download Visual Resume",
                    data=img_buffer.getvalue(),
                    file_name=f"Visual_Resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key="download_visual_resume"
                )

        with anal_tab3:
            st.subheader("ATS (Applicant Tracking System) Analysis")
            jd = st.text_area(
                "Paste Job Description for ATS Analysis",
                key="analysis_job_desc_area",
                height=150,
                placeholder="Paste the job description here to check ATS compatibility..."
            )
            if jd:
                ats = ats_score(resume_text, jd)
                st.progress(ats/100, text=f"ATS Match Score: {ats}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ATS Compatibility", f"{ats}%")

                with col2:
                    if st.button("Save This Analysis", key="analysis_save_btn"):
                        resume_data = {
                            "date": datetime.now().isoformat(),
                            "category": category,
                            "score": score,
                            "ats_score": ats,
                            "skills_count": sum(len(v) for v in skills.values())
                        }
                        auth.add_resume_to_user(st.session_state.username, resume_data)
                        st.success("✅ Analysis saved to your history!")
                        st.balloons()

                # ATS Tips
                st.subheader("ATS Optimization Tips")
                if ats < 50:
                    st.warning("**Low ATS Match** - Your resume needs significant optimization")
                    st.write("""
                    1. **Add more keywords** from the job description
                    2. **Use the exact terminology** mentioned in the job posting
                    3. **Include specific technologies** and tools mentioned
                    4. **Reformat your resume** to be ATS-friendly (no tables, fancy formatting)
                    """)
                elif ats < 80:
                    st.info("**Good ATS Match** - Some improvements possible")
                    st.write("""
                    1. **Verify all required skills** are mentioned
                    2. **Add quantifiable achievements** with numbers
                    3. **Check for missing keywords** from the job description
                    """)
                else:
                    st.success("**Excellent ATS Match** - Your resume is well-optimized!")
                    st.write("Keep up the good work! Your resume should pass through most ATS systems.")
            else:
                st.info("Paste a job description above to check ATS compatibility.")

        with anal_tab4:
            st.subheader("AI-Powered Suggestions")
            tips = resume_suggestions(score, skills)
            if tips:
                for i, tip in enumerate(tips):
                    st.markdown(f'<div class="card floating">**{i+1}.** {tip}</div>', unsafe_allow_html=True)
            else:
                st.success("Great! No major improvements needed.")

            # Template suggestion based on role
            st.subheader("💡 Template Recommendation")
            if "Academic" in category or "Entry" in category or "Student" in resume_text.lower():
                st.write("**Recommended Template:** Academic/Entry Level")
                st.write("This template highlights education, projects, and skills effectively for entry-level positions.")
            elif "Manager" in category or "Lead" in category or "Director" in category:
                st.write("**Recommended Template:** Modern Professional")
                st.write("This template emphasizes leadership, achievements, and professional experience.")
            elif "Analyst" in category or "Scientist" in category:
                st.write("**Recommended Template:** ATS Optimized")
                st.write("This template is keyword-rich and optimized for technical roles.")
            else:
                st.write("**Recommended Template:** Clean Minimalist")
                st.write("This template works well for most professional roles with a clean, organized layout.")

# =========================
# Render Templates Tab
# =========================
def render_templates_tab():
    """Render the Templates tab with enhanced features"""
    st.subheader("📋 Resume Templates & Generators")

    # Template selection
    template_choice = st.radio(
        "Choose a template style:",
        list(RESUME_TEMPLATES.keys()),
        key="templates_radio"
    )

    # Output format selection
    output_format = st.selectbox(
        "Select Output Format:",
        ["Preview Only", "PDF", "HTML", "Image", "Text"],
        key="templates_output_format"
    )

    # Profile image upload (optional)
    profile_image = None
    with st.expander("🖼️ Optional: Add Profile Image"):
        profile_image = st.file_uploader(
            "Upload Profile Photo (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="templates_profile_image"
        )
        if profile_image:
            st.image(profile_image, caption="Profile Image", width=150)

    # Manual data input for templates
    with st.expander("📝 Manual Data Input (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            manual_name = st.text_input("Name", key="manual_name")
            manual_email = st.text_input("Email", key="manual_email")
            manual_phone = st.text_input("Phone", key="manual_phone")
        with col2:
            manual_location = st.text_input("Location", key="manual_location")
            manual_linkedin = st.text_input("LinkedIn URL", key="manual_linkedin")
            manual_category = st.text_input("Job Category", key="manual_category")

    # Generate button
    if st.button("🚀 Generate Resume", type="primary", key="generate_resume_btn"):
        # Prepare data
        personal_info = {
            "name": manual_name if manual_name else "[YOUR NAME]",
            "email": manual_email if manual_email else "[Email]",
            "phone": manual_phone if manual_phone else "[Phone]",
            "location": manual_location if manual_location else "[City, State]",
            "linkedin": manual_linkedin if manual_linkedin else "[LinkedIn URL]",
            "github": "[GitHub URL]",
            "portfolio": "[Portfolio URL]",
            "summary": "[Professional Summary]"
        }

        # Create sample data
        experience = ["• [Job Title], [Company Name] | [City, State] | [Start Date] - [End Date]"]
        education = ["• [Degree Name], [Major]", "• [University Name], [City, State]"]
        skills = {"Programming": ["Python", "Java"], "Tools": ["Git", "Docker"]}
        projects = ["[Project Name]", "• [Project Description]"]
        certifications = ["• [Certification Name], [Issuing Organization]"]
        category = manual_category if manual_category else "Software Developer"

        # Generate based on output format
        if output_format == "PDF":
            # Handle profile image
            profile_img_data = None
            if profile_image:
                profile_img_data = io.BytesIO(profile_image.getvalue())

            # Generate PDF
            pdf_data = template_engine.create_resume_pdf(
                template_choice,
                personal_info,
                experience,
                education,
                skills,
                projects,
                certifications,
                category,
                profile_img_data
            )

            # Download button
            st.download_button(
                label="📥 Download PDF Resume",
                data=pdf_data,
                file_name=f"Resume_{personal_info['name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                key="templates_download_pdf"
            )
            st.success("PDF generated successfully!")

        elif output_format == "HTML":
            # Generate HTML
            html_content = template_engine.create_html_resume(
                template_choice,
                personal_info,
                experience,
                education,
                skills,
                category
            )

            # Download button
            st.download_button(
                label="📥 Download HTML Resume",
                data=html_content,
                file_name=f"Resume_{personal_info['name'].replace(' ', '_')}.html",
                mime="text/html",
                key="templates_download_html"
            )

            # Preview
            with st.expander("HTML Preview", expanded=True):
                st.components.v1.html(html_content, height=500, scrolling=True)

        elif output_format == "Image":
            # Generate visual resume
            img_buffer = template_engine.generate_visual_resume(
                f"{personal_info['name']} {personal_info['email']}",
                category,
                skills
            )

            # Download button
            st.download_button(
                label="📥 Download Resume Image",
                data=img_buffer,
                file_name=f"Resume_Visual_{datetime.now().strftime('%Y%m%d')}.png",
                mime="image/png",
                key="templates_download_img"
            )

            # Preview
            st.image(img_buffer, caption="Generated Visual Resume")

        elif output_format == "Text":
            # Show text template
            st.text_area(
                "Generated Resume Text",
                RESUME_TEMPLATES[template_choice],
                height=400,
                key="templates_text_output"
            )

            # Download button
            st.download_button(
                label="📥 Download Text Resume",
                data=RESUME_TEMPLATES[template_choice],
                file_name=f"{template_choice.replace(' ', '_')}_Resume.txt",
                mime="text/plain",
                key="templates_download_text"
            )
        else:
            # Preview only
            st.text_area(
                f"{template_choice} Template",
                RESUME_TEMPLATES[template_choice],
                height=400,
                key="templates_preview_only"
            )

    # Template preview section
    st.markdown("---")
    st.subheader("📄 Template Preview")
    st.text_area(
        "Template Content",
        RESUME_TEMPLATES[template_choice],
        height=300,
        key="templates_final_preview"
    )

# =========================
# Render History Tab
# =========================
def render_history_tab():
    """Render the History tab"""
    st.subheader("📈 My Analysis History")

    if st.session_state.username in auth.users and 'resumes' in auth.users[st.session_state.username]:
        resumes = auth.users[st.session_state.username]['resumes']
        if resumes:
            st.write(f"Total analyses saved: **{len(resumes)}**")

            # Display recent analyses
            for i, resume in enumerate(reversed(resumes[-5:])):
                index = len(resumes) - i
                with st.expander(f"Analysis #{index} - {resume['date'][:10]}", expanded=(i==0)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Category", resume['category'])
                    with col2:
                        st.metric("Score", f"{resume['score']}/100")
                    with col3:
                        st.metric("ATS", f"{resume.get('ats_score', 'N/A')}")
                    with col4:
                        st.metric("Skills", resume.get('skills_count', 0))

                    # Quick actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"View Details #{index}", key=f"view_details_{i}"):
                            st.session_state.selected_history = resume
                            st.info(f"Showing details for Analysis #{index}")

                    with col2:
                        if st.button(f"Delete #{index}", key=f"delete_{i}"):
                            # Remove from list
                            rev_index = resumes.index(resume)
                            del auth.users[st.session_state.username]['resumes'][rev_index]
                            auth.save_users()
                            st.success("✅ Analysis deleted!")
                            st.rerun()
                            st.balloons()

            # Show stats
            if len(resumes) > 1:
                st.subheader("📊 Your Progress")
                avg_score = sum(r['score'] for r in resumes) / len(resumes)
                max_score = max(r['score'] for r in resumes)
                min_score = min(r['score'] for r in resumes)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Score", f"{avg_score:.1f}/100")
                with col2:
                    st.metric("Best Score", f"{max_score}/100")
                with col3:
                    st.metric("First Score", f"{min_score}/100")
                    st.markdown("*(Score from your first saved analysis)*")

                # Progress chart suggestion
                if max_score > min_score:
                    st.success("**🎉 Great progress!** Your resume scores are improving.")
                else:
                    st.info("**💪 Keep optimizing!** Use the suggestions to improve your resume.")
        else:
            st.info("No saved analyses yet. Analyze a resume in the 'Resume Analysis' tab to see your history here.")
    else:
        st.info("No history available. Please analyze a resume first.")
        st.session_state.current_tab = "analysis"

# =========================
# Smart Navigation Component
# =========================
def smart_navigation(current_tab):
    """Create smart navigation bar"""

    st.markdown('<div class="smart-nav">', unsafe_allow_html=True)

    nav_cols = st.columns(3)

    # Analysis Tab
    with nav_cols[0]:
        is_active = current_tab == "analysis"
        css_class = "nav-item active" if is_active else "nav-item"
        if st.button("📊 Resume Analysis", key="nav_analysis_main", use_container_width=True):
            st.session_state.current_tab = "analysis"
            st.rerun()
            st.balloons()

    # Templates Tab
    with nav_cols[1]:
        is_active = current_tab == "templates"
        css_class = "nav-item active" if is_active else "nav-item"
        if st.button("📋 Templates", key="nav_templates_main", use_container_width=True):
            st.session_state.current_tab = "templates"
            st.rerun()
            st.balloons()

    # History Tab
    with nav_cols[2]:
        is_active = current_tab == "history"
        css_class = "nav-item active" if is_active else "nav-item"
        if st.button("📈 History", key="nav_history_main", use_container_width=True):
            st.session_state.current_tab = "history"
            st.rerun()
            st.balloons()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Main Dashboard
# =========================
def main_dashboard():
    st.set_page_config("AI Resume Analyzer", "📄", layout="wide")

    # Initialize session state with safe defaults
    if 'selected_template' not in st.session_state:
        st.session_state.selected_template = "Modern Professional"
    if 'extracted_info' not in st.session_state:
        st.session_state.extracted_info = {}
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "analysis"
    if 'username' not in st.session_state:
        st.session_state.username = "Guest"
    if 'show_celebration' not in st.session_state:
        st.session_state.show_celebration = True

    # Sidebar with unique keys
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {st.session_state.username}!")

        # Celebration on first load
        if st.session_state.get('show_celebration'):
            st.balloons()
            st.session_state.show_celebration = False

        mode = st.radio("🌗 Theme Mode", ["Light", "Dark"], key="dashboard_theme_mode")
        accent = st.color_picker("🎨 Accent Color", "#2563eb", key="dashboard_accent_color")
        load_css(mode, accent)

        # Quick navigation in sidebar
        st.markdown("### 🚀 Quick Actions")
        if st.button("📊 New Analysis", key="dashboard_new_analysis", use_container_width=True):
            st.session_state.current_tab = "analysis"
            st.rerun()
            st.balloons()

        if st.button("📋 Generate Resume", key="dashboard_generate_resume", use_container_width=True):
            st.session_state.current_tab = "templates"
            st.rerun()
            st.balloons()

        if st.button("📈 Check History", key="dashboard_check_history", use_container_width=True):
            st.session_state.current_tab = "history"
            st.rerun()
            st.balloons()

        st.markdown("---")

        if st.button("🚪 Logout", key="dashboard_logout_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            st.success("Logged out successfully!")
            st.balloons()

    # Main Content
    st.title("📄 AI Resume Analyzer Dashboard")
    st.markdown("Analyze your resume and generate professional templates in multiple formats!")

    # Smart Navigation Bar
    smart_navigation(st.session_state.current_tab)

    # Tab switching animation effect
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # Render appropriate tab based on navigation
    if st.session_state.current_tab == "analysis":
        render_analysis_tab()
    elif st.session_state.current_tab == "templates":
        render_templates_tab()
    elif st.session_state.current_tab == "history":
        render_history_tab()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Main Application
# =========================
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Marquee Disclaimer with animation
    st.markdown("""
    <style>
    .marquee-container {
        position: sticky;
        top: 0;
        z-index: 9999;
        width: 100%;
        overflow: hidden;
        background: linear-gradient(90deg, #020617, #020617);
        padding: 12px 0;
        border-bottom: 2px solid #2563eb;
    }
    .marquee-text {
        display: inline-block;
        white-space: nowrap;
        animation: scroll-left 18s linear infinite;
        font-size: 18px;
        font-weight: 600;
        color: #38bdf8;
        padding-left: 100%;
    }
    @keyframes scroll-left {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-100%); }
    }
    </style>
    <div class="marquee-container">
                <div class="marquee-text">
            ⚠️ AI Resume Analyzer - For educational purposes only. Use auto-fill to create optimized resumes from your existing resume.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Route to appropriate page
    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()