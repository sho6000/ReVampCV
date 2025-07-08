import time
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import base64
from PIL import Image
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import regex as re
import textstat

# Document processing libraries
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.layout import LAParams
import docx

# NLP libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="ReVamp CV",
    # page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App description
st.markdown("""
### AI-powered resume analyzer and enhancer

ReVamp CV helps job seekers optimize their resumes for specific job descriptions. 
Upload your resume and a job description to receive AI-powered suggestions, skill gap analysis, 
and tailored content improvements to increase your chances of getting past ATS systems and catching recruiters' attention.
The main goal of this project is to create a smart, AI-powered web application called
ReVamp CV. This tool helps job seekers customize and improve their resumes based on
specific job descriptions (JD). In today's competitive job market, generic resumes often do
not grab the attention of recruiters and are likely to be rejected by automated Applicant
Tracking Systems (ATS). This application tackles that issue by giving users a smart tool
to tailor their resumes to better fit what employers expect.
""")

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    st.warning("Please download the spaCy model by running: python -m spacy download en_core_web_sm")
    nlp = spacy.blank('en')

# Helper functions for file processing
def extract_text_from_pdf(pdf_file):
    try:
        # pdfminer.six provides better text extraction than PyPDF2
        # Configure layout parameters for better text extraction
        laparams = LAParams(
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            boxes_flow=0.5
        )
        
        # Change this line from pdf_extract_text to pdf_extract
        text = pdf_extract_text(pdf_file, laparams=laparams)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text(file):
    file_ext = os.path.splitext(file.name)[1].lower()
    
    if file_ext == ".pdf":
        return extract_text_from_pdf(file)
    elif file_ext == ".docx":
        return extract_text_from_docx(file)
    elif file_ext == ".txt":
        return file.getvalue().decode("utf-8")
    else:
        st.error(f"Unsupported file format: {file_ext}")
        return ""

def calculate_similarity_score(resume_text, jd_text):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform the texts to TF-IDF vectors
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100  # Convert to percentage
    except:
        st.error("Error calculating similarity score. Please check your resume and job description.")
        return 0

def summarize_text(text, max_length=500):
    """Generate a simple summary of the text using spaCy"""
    doc = nlp(text)
    
    # Get the sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # If there are fewer than 3 sentences, return the original text
    if len(sentences) <= 3:
        return text
    
    # Otherwise, take the first 3 sentences as a summary
    summary = " ".join(sentences[:3])
    
    # If the summary is still too long, truncate it
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
        
    return summary

def extract_key_points(text):
    """Extract key points from job description text"""
    doc = nlp(text)
    
    # Get the sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # If there are very few sentences, use them all
    if len(sentences) <= 5:
        return sentences
    
    # Otherwise, take key sentences based on position and keywords
    important_keywords = ['required', 'skill', 'qualification', 'experience', 'responsibilities', 
                         'duties', 'knowledge', 'abilities', 'proficient', 'familiar', 'degree',
                         'years', 'background', 'expertise']
    
    key_points = []
    
    # Always include the first sentence as it often contains the role/position
    if sentences[0]:
        key_points.append(sentences[0])
    
    # Add sentences that contain important keywords
    for sent in sentences:
        if any(keyword in sent.lower() for keyword in important_keywords):
            if sent not in key_points and len(sent) > 15:  # Avoid duplicates and very short sentences
                key_points.append(sent)
    
    # If we still don't have enough points, add more sentences from the beginning
    if len(key_points) < 5:
        for sent in sentences[1:6]:  # Skip first sentence as we already added it
            if sent not in key_points and len(sent) > 15:
                key_points.append(sent)
    
    # Limit to 8 key points
    return key_points[:8]

# Define skill lists at module level so they're accessible throughout the app
tech_skills = [
    "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "php", "swift",
    "kotlin", "typescript", "html", "css", "sql", "nosql", "mongodb", "mysql", "postgresql",
    "oracle", "react", "angular", "vue", "node.js", "express", "django", "flask", "spring",
    "asp.net", "jquery", "bootstrap", "tailwind", "aws", "azure", "gcp", "firebase",
    "docker", "kubernetes", "jenkins", "ci/cd", "git", "github", "bitbucket",
    "machine learning", "deep learning", "ai", "data science", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "data analysis", "data visualization", "power bi",
    "tableau", "excel", "vba", "sap", "salesforce", "jira", "confluence", "agile", "scrum"
]

soft_skills = [
    "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "time management", "project management", "analytical", "detail oriented", "creativity",
    "adaptability", "flexibility", "organization", "planning", "decision making",
    "conflict resolution", "presentation", "negotiation", "customer service", "interpersonal",
    "research", "writing", "editing", "public speaking", "mentoring", "coaching", "training"
]

# Then modify the extract_skills function to use these global lists
def extract_skills(text):
    """Extract skills from text using keyword matching"""
    # Use the global skill lists
    global tech_skills, soft_skills
    
    # Combine all skills
    all_skills = tech_skills + soft_skills
    
    # Find skills in the text
    found_skills = []
    text_lower = text.lower()
    
    for skill in all_skills:
        if skill in text_lower:
            found_skills.append(skill)
    
    return found_skills

# Main app function with Resume Enhancement section removed
def main():
    st.markdown("---")
    
    # File upload section
    st.subheader("Upload Your Files")
    col1, col2 = st.columns(2)
    
    with col1:
        resume_file = st.file_uploader("Choose your resume file", type=["pdf", "docx", "txt"])
    
    with col2:
        jd_file = st.file_uploader("Choose the job description file", type=["pdf", "docx", "txt"])
    
    # Process button and results - only show button when both files are present
    if resume_file is not None and jd_file is not None:
        # Show a full-width button
        analyze_button = st.button("Analyze Resume", use_container_width=True)
        
        # Only proceed with analysis when the button is clicked
        if analyze_button:
            with st.spinner("Processing your files..."):
                # Extract text from files
                resume_text = extract_text(resume_file)
                jd_text = extract_text(jd_file)
                
                # Calculate similarity score
                similarity_score = calculate_similarity_score(resume_text, jd_text)
                
                # Generate JD summary
                jd_summary = summarize_text(jd_text)
                
                # Extract key points from JD
                jd_key_points = extract_key_points(jd_text)
                
                # Extract skills from both resume and JD
                resume_skills = extract_skills(resume_text)
                jd_skills = extract_skills(jd_text)
                
                # Find missing skills
                missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
                
                # Store in session state for persistence
                st.session_state.resume_text = resume_text
                st.session_state.jd_text = jd_text
                st.session_state.similarity_score = similarity_score
                st.session_state.jd_summary = jd_summary
                st.session_state.jd_key_points = jd_key_points
                st.session_state.resume_skills = resume_skills
                st.session_state.jd_skills = jd_skills
                st.session_state.missing_skills = missing_skills
                
            # Display results
            st.subheader("Analysis Results")
            
            # Display match level message
            if similarity_score < 50:
                st.error("Your resume needs significant improvements to match this job description.")
            elif similarity_score < 70:
                st.warning("Your resume could use some targeted improvements for this job.")
            else:
                st.success("Your resume is well-matched to this job description!")
                
            # Additional visualization - Show key stats
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                st.metric(
                    label="Match Score", 
                    value=f"{similarity_score:.1f}%",
                    delta=f"{similarity_score - 50:.1f}%" if similarity_score > 50 else f"{similarity_score - 50:.1f}%",
                    delta_color="normal"
                )

            with stats_col2:
                # Calculate word count for resume
                resume_word_count = len(st.session_state.resume_text.split())
                st.metric(label="Resume Word Count", value=resume_word_count)

            with stats_col3:
                # Simple readability metric based on sentence length
                try:
                    sentences = st.session_state.resume_text.split('.')
                    words = st.session_state.resume_text.split()
                    avg_sentence_length = len(words) / max(len(sentences), 1)
                    # Simple heuristic: shorter sentences = better readability
                    readability_score = max(0, min(100, (25 - avg_sentence_length) * 4))
                    st.metric(label="Resume Readability", value=f"{readability_score:.1f}/100")
                except:
                    st.metric(label="Resume Readability", value="N/A")

            st.progress(similarity_score/100)
            
            # Job Description Summary expandable section
            with st.expander("Job Description Summary"):
                st.markdown("### Key Points from the Job Description")
                
                # Display key points as a bulleted list
                for point in st.session_state.jd_key_points:
                    st.markdown(f"• {point}")
            
            # Missing Skills expandable section
            with st.expander("Skills Analysis"):
                # Create tabs for different views of skills analysis
                skill_tabs = st.tabs(["Missing Skills", "Resume Skills", "Skills Matching", "Resume Optimizer"])
                
                with skill_tabs[0]:  # Missing Skills tab
                    if st.session_state.missing_skills:
                        st.markdown("### Skills Missing in Your Resume")
                        st.markdown("These skills are mentioned in the job description but not found in your resume:")
                        
                        # Group missing skills by category
                        missing_tech_skills = [skill for skill in st.session_state.missing_skills 
                                            if skill in tech_skills]
                        missing_soft_skills = [skill for skill in st.session_state.missing_skills 
                                            if skill in soft_skills]
                        
                        # Create columns for better layout
                        missing_col1, missing_col2 = st.columns([1, 1])
                        
                        # Display missing skills by category in columns
                        with missing_col1:
                            if missing_tech_skills:
                                st.markdown("#### Technical Skills")
                                # Improved styling with better contrast
                                for skill in missing_tech_skills:
                                    st.markdown(f"""
                                    <div style='padding: 8px 12px; margin: 5px 0; background-color: #e6f0fa; border-radius: 5px; border: 1px solid #c0d6e9;'>
                                        <span style='font-size: 16px; color: #0a2540;'>⚙️ <b>{skill}</b></span>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        with missing_col2:
                            if missing_soft_skills:
                                st.markdown("#### Soft Skills")
                                # Improved styling with better contrast
                                for skill in missing_soft_skills:
                                    st.markdown(f"""
                                    <div style='padding: 8px 12px; margin: 5px 0; background-color: #f5f0fa; border-radius: 5px; border: 1px solid #d8c0e9;'>
                                        <span style='font-size: 16px; color: #0a2540;'>👥 <b>{skill}</b></span>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Priority score for missing skills (based on frequency in JD)
                        st.markdown("### Priority Skills to Add")
                        st.markdown("These missing skills appear most frequently in the job description:")
                        
                        # Count occurrences of each missing skill in the JD
                        skill_priorities = {}
                        jd_lower = st.session_state.jd_text.lower()
                        
                        for skill in st.session_state.missing_skills:
                            count = jd_lower.count(skill)
                            skill_priorities[skill] = count
                        
                        # Sort by count (descending)
                        sorted_priorities = sorted(skill_priorities.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)
                        
                        # Display top 5 priority skills with improved styling
                        priority_cols = st.columns(5)
                        for i, (skill, count) in enumerate(sorted_priorities[:5]):
                            with priority_cols[i]:
                                st.markdown(f"""
                                <div style='text-align: center; padding: 12px; background-color: #f0f7ff; border-radius: 5px; height: 100%; border: 1px solid #c0d6e9;'>
                                    <div style='font-size: 18px; font-weight: bold; color: #0a2540;'>{skill}</div>
                                    <div style='font-size: 14px; color: #486581;'>Mentioned {count} times</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Updated Skill Improvement Suggestions section with better text contrast

                        st.markdown("### Skill Improvement Suggestions")
                        st.markdown("To improve your resume, try to incorporate these missing skills in relevant sections:")

                        # Example suggestions with better formatting and improved text contrast
                        st.markdown("""
                        <div style='padding: 15px; background-color: #f8f9fa; border-left: 5px solid #4CAF50; margin-bottom: 10px;'>
                            <p style='color: #333333; margin: 5px 0;'><b>1. Add missing skills</b> to your <b>Skills</b> section</p>
                            <p style='color: #333333; margin: 5px 0;'><b>2. Incorporate these skills</b> in your <b>Work Experience</b> descriptions</p>
                            <p style='color: #333333; margin: 5px 0;'><b>3. Mention relevant skills</b> in your <b>Summary</b> or <b>Objective</b> statement</p>
                            <p style='color: #333333; margin: 5px 0;'><b>4. Highlight projects</b> demonstrating these skills in a <b>Projects</b> section</p>
                            <p style='color: #333333; margin: 5px 0;'><b>5. Consider online courses</b> to develop these skills further</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("Great job! Your resume already contains all the key skills mentioned in the job description.")
                
                with skill_tabs[1]:  # Resume Skills tab
                    st.markdown("### Skills Found in Your Resume")
                    
                    # Group resume skills by category
                    resume_tech_skills = [skill for skill in st.session_state.resume_skills 
                                         if skill in tech_skills]
                    resume_soft_skills = [skill for skill in st.session_state.resume_skills 
                                         if skill in soft_skills]
                    
                    # Calculate statistics
                    total_skills = len(st.session_state.resume_skills)
                    tech_percent = len(resume_tech_skills) / total_skills * 100 if total_skills > 0 else 0
                    soft_percent = len(resume_soft_skills) / total_skills * 100 if total_skills > 0 else 0
                    
                    # Display skill distribution chart
                    st.markdown("#### Skill Distribution")
                    
                    # Create consistent height and style for the chart
                    st.markdown("""
                    <style>
                    .stChart > div > div > svg {
                        height: 300px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    skill_chart_data = pd.DataFrame({
                        'Category': ['Technical Skills', 'Soft Skills'],
                        'Percentage': [tech_percent, soft_percent]
                    })
                    st.bar_chart(skill_chart_data.set_index('Category'))
                    
                    # Create container for skills display
                    skills_container = st.container()
                    
                    with skills_container:
                        # Display skills in a more organized grid with better styling
                        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
                        
                        if resume_tech_skills:
                            st.markdown("#### Technical Skills")
                            # Create a grid layout
                            cols = st.columns(3)
                            for i, skill in enumerate(resume_tech_skills):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    if skill in st.session_state.jd_skills:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #e6f4ea; border-radius: 5px; border: 1px solid #b7dfb9;'>
                                            <span style='font-size: 16px; color: #0a2540;'>✅ <b>{skill}</b></span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #f0f2f6; border-radius: 5px; border: 1px solid #d0d7e2;'>
                                            <span style='font-size: 16px; color: #0a2540;'>ℹ️ {skill}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
                        
                        if resume_soft_skills:
                            st.markdown("#### Soft Skills")
                            # Create a grid layout
                            cols = st.columns(3)
                            for i, skill in enumerate(resume_soft_skills):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    if skill in st.session_state.jd_skills:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #e6f4ea; border-radius: 5px; border: 1px solid #b7dfb9;'>
                                            <span style='font-size: 16px; color: #0a2540;'>✅ <b>{skill}</b></span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #f0f2f6; border-radius: 5px; border: 1px solid #d0d7e2;'>
                                            <span style='font-size: 16px; color: #0a2540;'>ℹ️ {skill}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                
                with skill_tabs[2]:  # Skills Matching tab
                    st.markdown("### Skills Match Analysis")
                    
                    # Calculate matching statistics
                    total_jd_skills = len(st.session_state.jd_skills)
                    matching_skills = [skill for skill in st.session_state.resume_skills 
                                      if skill in st.session_state.jd_skills]
                    num_matching = len(matching_skills)
                    
                    # Create match percentage
                    match_percent = num_matching / total_jd_skills * 100 if total_jd_skills > 0 else 0
                    
                    # Display match statistics
                    st.metric(
                        label="Skills Match Rate", 
                        value=f"{match_percent:.1f}%",
                        help="Percentage of job description skills found in your resume"
                    )
                    
                    st.markdown(f"Your resume contains **{num_matching}** out of **{total_jd_skills}** skills mentioned in the job description.")
                    
                    # Display skill match visualization
                    match_data = {
                        'Status': ['Matching Skills', 'Missing Skills'],
                        'Count': [num_matching, total_jd_skills - num_matching]
                    }
                    match_df = pd.DataFrame(match_data)
                    
                    # Use a pie chart to visualize matching vs missing skills
                    fig = go.Figure(data=[go.Pie(
                        labels=match_df['Status'],
                        values=match_df['Count'],
                        hole=.3,
                        marker_colors=['#66BB6A', '#EF5350']
                    )])
                    fig.update_layout(title_text="Skills Match Breakdown")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display matching skills
                    if matching_skills:
                        st.markdown("#### Matching Skills")
                        st.markdown("These skills from your resume match the job description requirements:")
                        
                        match_cols = st.columns(3)
                        for i, skill in enumerate(matching_skills):
                            col_idx = i % 3
                            with match_cols[col_idx]:
                                st.markdown(f"✅ {skill}")
                
                with skill_tabs[3]:  # Resume Optimizer tab
                    display_optimizer_tab(st.session_state)
            

def display_optimizer_tab(st_session_state):
    """Advanced Resume Optimizer with Gemini AI"""
    st.markdown("### 🚀 AI-Powered Resume Optimizer")
    st.markdown("Enhance your resume content using Google's Gemini AI while maintaining honesty and authenticity.")
    
    # Check if API key is configured
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("⚠️ Gemini API key not configured. Please add your GOOGLE_API_KEY to the .env file.")
        return
    
    # Create optimization tabs
    opt_tabs = st.tabs(["AI Suggestions", "Content Optimizer", "Keyword Enhancement", "ATS Optimization"])
    
    with opt_tabs[0]:  # AI Suggestions
        st.markdown("#### 🤖 AI-Generated Improvement Suggestions")
        
        if st.button("Generate AI Suggestions", use_container_width=True):
            with st.spinner("Generating personalized suggestions with Gemini AI..."):
                suggestions = generate_resume_suggestions_with_gemini(
                    st_session_state.resume_text,
                    st_session_state.jd_text
                )
                
                # Store in session state to persist across page refreshes
                st.session_state.ai_suggestions = suggestions
        
        # Display suggestions if they exist in session state
        if hasattr(st.session_state, 'ai_suggestions') and st.session_state.ai_suggestions:
            st.markdown("### 📝 Personalized Recommendations")
            st.markdown(st.session_state.ai_suggestions)
            
            # Add a button to clear suggestions
            if st.button("Clear Suggestions"):
                if 'ai_suggestions' in st.session_state:
                    del st.session_state.ai_suggestions
                st.rerun()
    
    with opt_tabs[1]:  # Content Optimizer
        st.markdown("#### ✏️ AI-Powered Content Enhancement")
        
        # Section selector
        section_type = st.selectbox(
            "Select section to optimize:",
            ["experience", "summary", "skills", "education"]
        )
        
        # Text area for manual input
        current_content = st.text_area(
            f"Enter your current {section_type} section:",
            height=150,
            placeholder=f"Paste your {section_type} content here..."
        )
        
        if st.button("Optimize with AI", use_container_width=True):
            if current_content:
                with st.spinner("Optimizing content with Gemini AI..."):
                    optimized_content = optimize_resume_with_gemini(
                        current_content,
                        st_session_state.jd_text,
                        section_type
                    )
                    
                    # Store in session state
                    st.session_state.optimized_content = optimized_content
                    st.session_state.original_content = current_content
            else:
                st.warning("Please enter content to optimize")
        
        # Display optimized content if it exists in session state
        if hasattr(st.session_state, 'optimized_content') and st.session_state.optimized_content:
            st.markdown("### 📄 Optimized Content")
            st.markdown("**Original:**")
            st.info(st.session_state.original_content)
            
            st.markdown("**AI-Optimized:**")
            st.success(st.session_state.optimized_content)
            
            # Copy to clipboard button
            st.markdown("---")
            st.markdown("*Copy the optimized content above to use in your resume*")
            
            # Add a button to clear results
            if st.button("Clear Results"):
                if 'optimized_content' in st.session_state:
                    del st.session_state.optimized_content
                if 'original_content' in st.session_state:
                    del st.session_state.original_content
                st.rerun()
    
    with opt_tabs[2]:  # Keyword Enhancement
        st.markdown("#### 🔍 Strategic Keyword Enhancement")
        
        # Find missing keywords with context
        missing_keywords = find_contextual_keywords(
            st_session_state.resume_text,
            st_session_state.jd_text
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### High-Impact Keywords to Add")
            
            for keyword, context in missing_keywords[:5]:
                st.markdown(f"""
                <div style='padding: 10px; background-color: #f0f7ff; border-radius: 5px; margin: 5px 0;'>
                    <b>{keyword}</b><br>
                    <small style='color: #666;'>Context: {context}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### Suggested Placement")
            
            # Extract resume sections before suggesting keyword placement
            resume_sections = extract_resume_sections(st_session_state.resume_text)
            # Suggest where to place keywords
            placement_suggestions = suggest_keyword_placement(missing_keywords, resume_sections)
            
            # Replace nested expander with simple markdown containers
            for section, keywords in placement_suggestions.items():
                st.markdown(f"**Add to {section.title()} Section:**")
                for keyword in keywords:
                    st.markdown(f"• {keyword}")
                st.markdown("---")
    
    with opt_tabs[3]:  # ATS Optimization
        st.markdown("#### 🤖 ATS Compatibility Check")
        
        # Run ATS compatibility analysis
        ats_score, ats_issues = analyze_ats_compatibility(st_session_state.resume_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ATS Compatibility Score", f"{ats_score:.1f}%")
            
            # Show progress bar
            progress_color = "green" if ats_score > 80 else "orange" if ats_score > 60 else "red"
            st.progress(ats_score / 100)
        
        with col2:
            st.markdown("##### ATS Issues Found")
            
            if ats_issues:
                for issue in ats_issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ No major ATS issues detected!")
        
        # Provide ATS optimization suggestions
        st.markdown("##### ATS Optimization Recommendations")
        
        ats_recommendations = generate_ats_recommendations(ats_issues, st_session_state.resume_text)
        
        for i, rec in enumerate(ats_recommendations, 1):
            st.markdown(f"""
            <div style='padding: 12px; background-color: #f8f9fa; border-left: 4px solid #4CAF50; margin: 10px 0;'>
                <b>{i}. {rec['title']}</b><br>
                {rec['description']}
            </div>
            """, unsafe_allow_html=True)

# Helper functions for the optimizer

def extract_resume_sections(resume_text):
    """Extract different sections from resume using regex and NLP"""
    sections = {}
    
    # Common section headers
    section_patterns = {
        'experience': r'(?i)(experience|employment|work history|professional experience)(.*?)(?=(?:education|skills|projects|certifications|$))',
        'skills': r'(?i)(skills|technical skills|competencies)(.*?)(?=(?:experience|education|projects|certifications|$))',
        'education': r'(?i)(education|academic)(.*?)(?=(?:experience|skills|projects|certifications|$))',
        'summary': r'(?i)(summary|objective|profile)(.*?)(?=(?:experience|education|skills|projects|$))'
    }
    
    for section, pattern in section_patterns.items():
        match = re.search(pattern, resume_text, re.DOTALL)
        if match:
            sections[section] = match.group(2).strip()
    
    return sections

def extract_jd_requirements(jd_text):
    """Extract key requirements from job description"""
    doc = nlp(jd_text)
    
    requirements = []
    
    # Look for requirement indicators
    requirement_indicators = ['required', 'must have', 'should have', 'preferred', 'experience with']
    
    for sent in doc.sents:
        if any(indicator in sent.text.lower() for indicator in requirement_indicators):
            requirements.append(sent.text.strip())
    
    return requirements

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using sentence transformers"""
    if 'semantic_model' not in st.session_state:
        return 0
    
    try:
        embeddings = st.session_state.semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity * 100
    except:
        return 0

def find_matching_bullets(experience_text, jd_requirements, threshold=0.3):
    """Find bullet points that match JD requirements"""
    bullets = extract_bullet_points(experience_text)
    matching_bullets = []
    
    for bullet in bullets:
        for req in jd_requirements:
            similarity = calculate_semantic_similarity(bullet, req) / 100
            if similarity > threshold:
                matching_bullets.append(bullet)
                break
    
    return matching_bullets

def extract_bullet_points(text):
    """Extract bullet points from text"""
    lines = text.split('\n')
    bullets = []
    
    for line in lines:
        line = line.strip()
        if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
            bullets.append(line.lstrip('•-* '))
        elif re.match(r'^\d+\.', line):  # Numbered lists
            bullets.append(re.sub(r'^\d+\.\s*', '', line))
    
    return bullets

def calculate_bullet_relevance(bullet, jd_requirements):
    """Calculate how relevant a bullet point is to JD requirements"""
    if not jd_requirements:
        return 0
    
    max_similarity = 0
    for req in jd_requirements:
        similarity = calculate_semantic_similarity(bullet, req) / 100
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def optimize_bullet_point(bullet, jd_requirements):
    """Optimize a bullet point for better JD alignment while maintaining honesty"""
    # This is a simplified version - in production, you'd use T5/BART for paraphrasing
    
    # Extract action verbs and achievements
    action_verbs = ['developed', 'implemented', 'managed', 'led', 'created', 'improved', 'analyzed', 'designed']
    
    # Find the most relevant requirement
    best_req = ""
    best_similarity = 0
    
    for req in jd_requirements:
        similarity = calculate_semantic_similarity(bullet, req) / 100
        if similarity > best_similarity:
            best_similarity = similarity
            best_req = req
    
    # Simple optimization: enhance with stronger action verbs and quantifiers
    optimized = bullet
    
    # Replace weak verbs with stronger ones
    weak_to_strong = {
        'worked on': 'developed',
        'helped': 'contributed to',
        'did': 'executed',
        'made': 'created'
    }
    
    for weak, strong in weak_to_strong.items():
        if weak in optimized.lower():
            optimized = optimized.replace(weak, strong)
    
    # Add quantifiers if missing
    if not re.search(r'\d+', optimized):
        optimized = optimized + " (quantify with specific metrics when possible)"
    
    return optimized

def highlight_changes(original, optimized):
    """Highlight changes between original and optimized text"""
    # Simple diff highlighting
    if original == optimized:
        return "No changes made"
    
    return f"""
    <div style='background-color: #ffe6e6; padding: 5px; margin: 5px 0;'>
        <del>{original}</del>
    </div>
    <div style='background-color: #e6ffe6; padding: 5px; margin: 5px 0;'>
        <ins>{optimized}</ins>
    </div>
    """

def identify_weak_sections(resume_sections, jd_requirements):
    """Identify sections that need improvement"""
    weak_sections = {}
    
    for section, content in resume_sections.items():
        if content:
            score = calculate_semantic_similarity(content, ' '.join(jd_requirements))
            if score < 30:  # Low alignment threshold
                weak_sections[section] = [
                    "Add more relevant keywords from the job description",
                    "Include specific achievements and metrics",
                    "Align language with job requirements"
                ]
    
    return weak_sections

def find_contextual_keywords(resume_text, jd_text):
    """Find missing keywords with their context"""
    # Extract important terms from JD
    jd_doc = nlp(jd_text)
    resume_doc = nlp(resume_text)
    
    # Get important terms (nouns, adjectives that aren't in resume)
    jd_terms = set()
    resume_terms = set()
    
    for token in jd_doc:
        if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2:
            jd_terms.add(token.lemma_.lower())
    
    for token in resume_doc:
        if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2:
            resume_terms.add(token.lemma_.lower())
    
    missing_terms = jd_terms - resume_terms
    
    # Get context for missing terms
    contextual_keywords = []
    for term in list(missing_terms)[:10]:  # Limit to top 10
        # Find sentence containing the term in JD
        for sent in jd_doc.sents:
            if term in sent.text.lower():
                context = sent.text[:100] + "..." if len(sent.text) > 100 else sent.text
                contextual_keywords.append((term, context))
                break
    
    return contextual_keywords

def suggest_keyword_placement(missing_keywords, resume_sections):
    """Suggest where to place missing keywords"""
    placement = {}
    
    for keyword, context in missing_keywords:
        # Simple heuristic for placement
        if any(tech in keyword.lower() for tech in ['python', 'java', 'sql', 'aws']):
            if 'skills' not in placement:
                placement['skills'] = []
            placement['skills'].append(keyword)
        else:
            if 'experience' not in placement:
                placement['experience'] = []
            placement['experience'].append(keyword)
    
    return placement

def analyze_ats_compatibility(resume_text):
    """Analyze ATS compatibility issues"""
    issues = []
    score = 100
    
    # Check for common ATS issues
    if len(resume_text.split()) < 200:
        issues.append("Resume may be too short for proper ATS parsing")
        score -= 15
    
    if not re.search(r'(skills?|technical|competencies)', resume_text, re.IGNORECASE):
        issues.append("No clear skills section detected")
        score -= 20
    
    if not re.search(r'(experience|employment|work)', resume_text, re.IGNORECASE):
        issues.append("No clear experience section detected")
        score -= 25
    
    # Check for formatting issues
    if len(re.findall(r'[^\w\s]', resume_text)) > len(resume_text) * 0.1:
        issues.append("Too many special characters may confuse ATS")
        score -= 10
    
    return max(0, score), issues

def generate_ats_recommendations(issues, resume_text):
    """Generate ATS optimization recommendations"""
    recommendations = []
    
    if "too short" in str(issues):
        recommendations.append({
            'title': 'Expand Content',
            'description': 'Add more detailed descriptions of your achievements and responsibilities'
        })
    
    if "skills section" in str(issues):
        recommendations.append({
            'title': 'Add Skills Section',
            'description': 'Create a dedicated "Skills" or "Technical Skills" section with relevant keywords'
        })
    
    if "experience section" in str(issues):
        recommendations.append({
            'title': 'Organize Experience',
            'description': 'Create a clear "Professional Experience" or "Work Experience" section'
        })
    
    # Always include these best practices
    recommendations.extend([
        {
            'title': 'Use Standard Headers',
            'description': 'Use conventional section headers like "Experience", "Education", "Skills"'
        },
        {
            'title': 'Include Keywords',
            'description': 'Incorporate exact keywords from the job description naturally in your content'
        }
    ])
    
    return recommendations  # Add this missing return statement

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def optimize_resume_with_gemini(resume_text, jd_text, section_type="experience"):
    """Use Gemini to optimize resume content"""
    try:
        # Initialize Gemini model with the correct model name
        model = genai.GenerativeModel('gemini-2.0-flash')  # Changed from 'gemini-pro'
        
        # Create optimization prompt
        prompt = f"""
        As a professional resume writer, help optimize this resume section for the given job description.
        
        **Job Description:**
        {jd_text[:1000]}...
        
        **Current Resume Section ({section_type}):**
        {resume_text}
        
        **Instructions:**
        1. Rewrite the resume section to better align with the job description
        2. Use strong action verbs and quantifiable achievements
        3. Include relevant keywords from the job description naturally
        4. Maintain honesty - don't add false information
        5. Keep the same experience and skills, just improve the presentation
        6. Format as bullet points where appropriate
        
        **Optimized Resume Section:**
        """
        
        # Generate optimized content
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error optimizing with Gemini: {str(e)}")
        return "Error occurred during optimization"

def generate_resume_suggestions_with_gemini(resume_text, jd_text):
    """Generate improvement suggestions using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Changed from 'gemini-pro'
        
        prompt = f"""
        As an expert resume reviewer, analyze this resume against the job description and provide specific improvement suggestions.
        
        **Job Description:**
        {jd_text[:1000]}...
        
        **Resume:**
        {resume_text[:1500]}...
        
        **Please provide:**
        1. Top 5 specific improvements needed
        2. Missing keywords that should be added
        3. Suggested action verbs to use
        4. Recommendations for quantifying achievements
        5. Overall strategy for better alignment
        
        Format your response with clear sections and bullet points.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = f"Error generating suggestions: {str(e)}"
        st.error(error_msg)
        return error_msg
    
main()