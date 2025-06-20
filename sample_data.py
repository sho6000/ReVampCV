# Sample data for demonstration purposes

SAMPLE_JOB_DESCRIPTION = """
Senior Software Engineer

Job Description:
We are seeking a Senior Software Engineer to join our dynamic team. The ideal candidate will have strong experience in modern web development and cloud technologies.

Requirements:
- 5+ years of experience in software development
- Strong proficiency in Python, JavaScript, and React
- Experience with cloud platforms (AWS, Azure, or GCP)
- Knowledge of database systems (SQL and NoSQL)
- Experience with Docker and Kubernetes
- Strong problem-solving skills and attention to detail
- Excellent communication and teamwork abilities
- Bachelor's degree in Computer Science or related field

Responsibilities:
- Design and develop scalable web applications
- Collaborate with cross-functional teams
- Mentor junior developers
- Participate in code reviews and technical discussions
- Implement best practices and coding standards
- Troubleshoot and debug complex issues

Preferred Skills:
- Machine Learning and AI experience
- DevOps and CI/CD pipeline experience
- Microservices architecture
- Agile development methodologies
"""

SAMPLE_RESUME = """
JOHN DOE
Software Engineer
john.doe@email.com | (555) 123-4567 | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 4 years of experience in web development and cloud technologies. Passionate about creating scalable solutions and mentoring team members.

TECHNICAL SKILLS
Programming Languages: Python, JavaScript, Java, C++
Frontend: React, HTML5, CSS3, Bootstrap
Backend: Node.js, Django, Flask
Databases: MySQL, MongoDB, PostgreSQL
Cloud Platforms: AWS, Google Cloud Platform
Tools: Git, Docker, Jenkins, JIRA
Other: RESTful APIs, Microservices, Agile/Scrum

PROFESSIONAL EXPERIENCE

Software Engineer | TechCorp Inc. | 2021 - Present
- Developed and maintained web applications using React and Node.js
- Implemented RESTful APIs and microservices architecture
- Collaborated with cross-functional teams in an Agile environment
- Mentored 2 junior developers and conducted code reviews
- Reduced application load time by 40% through optimization

Junior Developer | StartupXYZ | 2019 - 2021
- Built responsive web applications using Python and Django
- Worked with MySQL and MongoDB databases
- Participated in daily stand-ups and sprint planning
- Contributed to the development of CI/CD pipelines

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2015 - 2019
GPA: 3.8/4.0

PROJECTS
E-commerce Platform (2023)
- Built a full-stack e-commerce application using React and Node.js
- Implemented payment processing and inventory management
- Deployed on AWS with Docker containers

Task Management App (2022)
- Developed a collaborative task management tool
- Used React for frontend and Python Flask for backend
- Integrated with MongoDB for data storage

CERTIFICATIONS
- AWS Certified Developer Associate
- Google Cloud Platform Certified
- Certified Scrum Master (CSM)

LANGUAGES
English (Native), Spanish (Intermediate)
"""

SAMPLE_ANALYSIS_RESULTS = {
    "overall_score": 75.0,
    "matched_skills": [
        "python", "javascript", "react", "aws", "docker", "mysql", "mongodb",
        "agile", "git", "restful apis", "microservices"
    ],
    "missing_skills": [
        "kubernetes", "azure", "machine learning", "ai", "devops", "ci/cd"
    ],
    "experience_score": 80.0,
    "education_score": 90.0,
    "recommendations": [
        "Add Kubernetes experience to your skills section",
        "Include Azure cloud platform experience",
        "Highlight any machine learning or AI projects",
        "Add more DevOps and CI/CD pipeline experience",
        "Emphasize leadership and mentoring experience"
    ],
    "suggested_improvements": [
        "Add a section highlighting Kubernetes experience",
        "Include Azure cloud platform in technical skills",
        "Add machine learning projects to showcase AI experience",
        "Expand on DevOps practices and CI/CD experience",
        "Quantify achievements with specific metrics"
    ]
}

def get_sample_data():
    """Return sample data for demonstration"""
    return {
        "job_description": SAMPLE_JOB_DESCRIPTION,
        "resume": SAMPLE_RESUME,
        "analysis_results": SAMPLE_ANALYSIS_RESULTS
    } 