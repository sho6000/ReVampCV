import streamlit as st
from resume_analyzer import ResumeAnalyzer

def create_header():
    """Create the main header for the application"""
    st.markdown("""
    <div class="main-header">
        <h1>📄 Resume Analyzer Pro</h1>
        <p>AI-Powered Resume Analysis & Optimization Tool</p>
        <p>Upload your resume and job description to get instant feedback and improvements</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar with additional features"""
    with st.sidebar:
        st.markdown("## 🛠️ Features")
        st.markdown("""
        - **📊 Smart Analysis**: AI-powered resume analysis
        - **🎯 Skills Matching**: Compare your skills with job requirements
        - **💡 Recommendations**: Get actionable improvement suggestions
        - **✨ Auto-Improvement**: Generate enhanced resume versions
        - **📋 Side-by-Side Comparison**: View original vs improved resume
        """)
        
        st.markdown("## 📈 How it Works")
        st.markdown("""
        1. **Upload** your resume and job description
        2. **Analyze** the match using AI
        3. **Review** detailed feedback and scores
        4. **Improve** your resume with suggestions
        5. **Download** the enhanced version
        """)
        
        st.markdown("## 🔧 Supported Formats")
        st.markdown("""
        - **Resume**: PDF, DOCX, TXT
        - **Job Description**: PDF, DOCX, TXT
        """)
        
        # Add a demo section
        st.markdown("## 🎯 Quick Demo")
        if st.button("Try Sample Analysis"):
            st.session_state.demo_mode = True
            st.success("Demo mode activated! Use sample data for testing.")

def create_upload_section(title: str, description: str, key: str):
    """Create a file upload section with modern styling"""
    st.markdown(f"""
    <div class="upload-section">
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=['pdf', 'docx', 'txt'],
        key=key,
        help="Upload your file here"
    )
    
    if uploaded_file is not None:
        try:
            # Initialize analyzer if not already done
            if 'resume_analyzer' not in st.session_state:
                st.session_state.resume_analyzer = ResumeAnalyzer()
            
            # Extract text from file
            content = st.session_state.resume_analyzer.extract_text_from_file(uploaded_file)
            
            # Show file info
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            st.info(f"📄 File size: {uploaded_file.size} bytes")
            
            return content
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return None
    
    return None

def create_analysis_section():
    """Create the analysis results section"""
    st.markdown("## 🔍 Analysis Results")
    
    # Placeholder for analysis results
    st.info("Upload both resume and job description to see analysis results.")

def create_metrics_display(metrics: dict):
    """Create a metrics display section"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Overall Score</h3>
            <h2>{metrics.get('overall_score', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🛠️ Skills Match</h3>
            <h2>{metrics.get('skills_score', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💼 Experience</h3>
            <h2>{metrics.get('experience_score', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎓 Education</h3>
            <h2>{metrics.get('education_score', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

def create_skills_comparison(matched_skills: list, missing_skills: list):
    """Create a skills comparison visualization"""
    st.markdown("### 🛠️ Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Skills Found in Resume**")
        if matched_skills:
            for skill in matched_skills:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
        else:
            st.info("No matching skills found")
    
    with col2:
        st.markdown("**❌ Missing Skills**")
        if missing_skills:
            for skill in missing_skills:
                st.markdown(f'<span class="missing-skill-tag">{skill}</span>', unsafe_allow_html=True)
        else:
            st.success("All required skills are present!")

def create_recommendations_section(recommendations: list):
    """Create a recommendations section"""
    st.markdown("### 💡 Recommendations")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    else:
        st.info("No specific recommendations available.")

def create_comparison_view(original_resume: str, improved_resume: str = None):
    """Create a side-by-side comparison view"""
    st.markdown("## 📋 Side-by-Side Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Original Resume")
        st.text_area("", value=original_resume, height=400, disabled=True)
    
    with col2:
        if improved_resume:
            st.markdown("### ✨ Improved Resume")
            st.text_area("", value=improved_resume, height=400, disabled=True)
        else:
            st.markdown("### 📝 Suggested Improvements")
            st.info("Generate improved resume to see the enhanced version here.")

def create_download_section(improved_resume: str):
    """Create a download section for the improved resume"""
    if improved_resume:
        st.markdown("### 📥 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download as TXT",
                data=improved_resume,
                file_name="improved_resume.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📋 Copy to Clipboard",
                data=improved_resume,
                file_name="improved_resume.txt",
                mime="text/plain",
                use_container_width=True
            )

def create_progress_indicator(step: int, total_steps: int = 4):
    """Create a progress indicator"""
    progress = step / total_steps
    
    st.markdown(f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>Step {step} of {total_steps}</span>
            <span>{progress*100:.0f}% Complete</span>
        </div>
        <div class="progress-bar" style="width: {progress*100}%;"></div>
    </div>
    """, unsafe_allow_html=True)

def create_error_message(error: str):
    """Create a styled error message"""
    st.markdown(f"""
    <div style="background: #ffebee; color: #c62828; padding: 1rem; border-radius: 10px; border-left: 5px solid #f44336;">
        <h4>❌ Error</h4>
        <p>{error}</p>
    </div>
    """, unsafe_allow_html=True)

def create_success_message(message: str):
    """Create a styled success message"""
    st.markdown(f"""
    <div style="background: #e8f5e8; color: #2e7d32; padding: 1rem; border-radius: 10px; border-left: 5px solid #4caf50;">
        <h4>✅ Success</h4>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True) 