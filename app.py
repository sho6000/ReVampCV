import streamlit as st
import os
import tempfile
from pathlib import Path
import base64
from resume_analyzer import ResumeAnalyzer
from ui_components import create_header, create_sidebar, create_upload_section, create_analysis_section
import plotly.graph_objects as go
import plotly.express as px
from sample_data import get_sample_data

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
        text-align: center;
    }
    
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .file-uploader {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    
    .comparison-container {
        display: flex;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .comparison-panel {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .skill-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .missing-skill-tag {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'resume_analyzer' not in st.session_state:
        st.session_state.resume_analyzer = ResumeAnalyzer()
    
    if 'job_description_content' not in st.session_state:
        st.session_state.job_description_content = None
    
    if 'resume_content' not in st.session_state:
        st.session_state.resume_content = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'improved_resume' not in st.session_state:
        st.session_state.improved_resume = None
    
    # Create header
    create_header()
    
    # Create sidebar
    create_sidebar()
    
    # Check for demo mode
    if st.session_state.get('demo_mode', False):
        sample_data = get_sample_data()
        st.session_state.job_description_content = sample_data['job_description']
        st.session_state.resume_content = sample_data['resume']
        st.session_state.analysis_results = sample_data['analysis_results']
        st.session_state.demo_mode = False
        st.success("Demo data loaded! You can now see the analysis results.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Job Description")
        job_description = create_upload_section(
            "Upload Job Description",
            "Upload the job description file (PDF, DOCX, TXT)",
            "job_description"
        )
        
        if job_description:
            st.session_state.job_description_content = job_description
            with st.expander("📄 Job Description Content", expanded=True):
                st.markdown(f"**Job Description:**")
                st.text_area("", value=job_description, height=300, disabled=True)
    
    with col2:
        st.markdown("### 📄 Resume")
        resume_content = create_upload_section(
            "Upload Resume",
            "Upload your resume file (PDF, DOCX, TXT)",
            "resume"
        )
        
        if resume_content:
            st.session_state.resume_content = resume_content
            with st.expander("📄 Resume Content", expanded=True):
                st.markdown(f"**Resume Content:**")
                st.text_area("", value=resume_content, height=300, disabled=True)
    
    # Analysis section
    if st.session_state.job_description_content and st.session_state.resume_content:
        st.markdown("---")
        st.markdown("## 🔍 Analysis Results")
        
        # Analyze button
        if st.button("🚀 Analyze Resume", use_container_width=True):
            with st.spinner("Analyzing your resume..."):
                try:
                    # Perform analysis
                    analysis_results = st.session_state.resume_analyzer.analyze_resume(
                        st.session_state.resume_content,
                        st.session_state.job_description_content
                    )
                    st.session_state.analysis_results = analysis_results
                    
                    # Display results
                    display_analysis_results(analysis_results)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    # Display previous results if available
    elif st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("## 🔍 Previous Analysis Results")
        display_analysis_results(st.session_state.analysis_results)

def display_analysis_results(results):
    """Display the analysis results in a comprehensive and visually appealing way"""
    
    # Overall score
    score = results.get('overall_score', 0)
    
    # Create score gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Resume Match Score"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Score interpretation
    if score >= 80:
        st.markdown("""
        <div class="success-card">
            <h3>🎉 Excellent Match!</h3>
            <p>Your resume is well-aligned with the job description. You have a strong chance of getting selected for this position.</p>
        </div>
        """, unsafe_allow_html=True)
    elif score >= 60:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ Good Match with Room for Improvement</h3>
            <p>Your resume shows good alignment but could be enhanced to better match the job requirements.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ Needs Significant Improvement</h3>
            <p>Your resume needs substantial updates to better align with the job description.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Detailed Analysis")
        
        # Skills analysis
        st.markdown("#### 🛠️ Skills Analysis")
        matched_skills = results.get('matched_skills', [])
        missing_skills = results.get('missing_skills', [])
        
        if matched_skills:
            st.markdown("**✅ Skills Found in Resume:**")
            for skill in matched_skills:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
        
        if missing_skills:
            st.markdown("**❌ Missing Skills:**")
            for skill in missing_skills:
                st.markdown(f'<span class="missing-skill-tag">{skill}</span>', unsafe_allow_html=True)
        
        # Experience analysis
        st.markdown("#### 💼 Experience Analysis")
        experience_score = results.get('experience_score', 0)
        st.markdown(f"**Experience Relevance:** {experience_score:.1f}%")
        
        # Education analysis
        st.markdown("#### 🎓 Education Analysis")
        education_score = results.get('education_score', 0)
        st.markdown(f"**Education Match:** {education_score:.1f}%")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        
        recommendations = results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
        else:
            st.markdown("No specific recommendations available.")
        
        # Generate improved resume
        if st.button("✨ Generate Improved Resume", use_container_width=True):
            with st.spinner("Generating improved resume..."):
                try:
                    improved_resume = st.session_state.resume_analyzer.generate_improved_resume(
                        st.session_state.resume_content,
                        st.session_state.job_description_content,
                        results
                    )
                    
                    st.session_state.improved_resume = improved_resume
                    
                    st.markdown("### 📝 Improved Resume")
                    st.text_area("Improved Resume Content", value=improved_resume, height=400)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Improved Resume",
                        data=improved_resume,
                        file_name="improved_resume.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating improved resume: {str(e)}")
    
    # Comparison view
    st.markdown("---")
    st.markdown("## 📋 Side-by-Side Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Original Resume")
        st.text_area("", value=st.session_state.resume_content, height=400, disabled=True)
    
    with col2:
        if st.session_state.improved_resume:
            st.markdown("### ✨ Improved Resume")
            st.text_area("", value=st.session_state.improved_resume, height=400, disabled=True)
        else:
            st.markdown("### 📝 Suggested Improvements")
            improvements = results.get('suggested_improvements', [])
            if improvements:
                for improvement in improvements:
                    st.markdown(f"• {improvement}")
            else:
                st.markdown("No specific improvements suggested.")

if __name__ == "__main__":
    main() 