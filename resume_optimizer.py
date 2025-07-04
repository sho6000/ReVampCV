# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go

# def display_optimizer_tab(st_session_state):
#     print("Resume optimizer tab is being called")
#     """Display the Resume Optimizer tab content"""
#     st.markdown("### Resume Optimization Recommendations")
    
#     # Calculate optimization statistics
#     total_missing = len(st_session_state.missing_skills)
#     total_jd_skills = len(st_session_state.jd_skills)
#     missing_percent = total_missing / total_jd_skills * 100 if total_jd_skills > 0 else 0
    
#     # Recommended sections to improve
#     st.markdown("#### 📋 Recommended Resume Improvements")
    
#     # Display optimization progress
#     st.progress(100 - missing_percent)
#     st.markdown(f"Your resume needs approximately **{missing_percent:.1f}%** improvement to fully match this job description.")
    
#     # Create columns for different optimization areas
#     opt_col1, opt_col2 = st.columns(2)
    
#     with opt_col1:
#         st.markdown("##### Content Recommendations")
        
#         # List of recommendations based on missing skills
#         if total_missing > 0:
#             st.markdown("""
#             <div style='padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef;'>
#                 <p style='color: #333333; margin: 5px 0;'><b>1. Skills Section:</b> Add missing technical and soft skills</p>
#                 <p style='color: #333333; margin: 5px 0;'><b>2. Work Experience:</b> Highlight achievements using job-specific keywords</p>
#                 <p style='color: #333333; margin: 5px 0;'><b>3. Professional Summary:</b> Include key skills and relevant experience</p>
#                 <p style='color: #333333; margin: 5px 0;'><b>4. Projects:</b> Showcase relevant projects that demonstrate missing skills</p>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.success("Your resume content is already well-matched to this job description!")
    
#     with opt_col2:
#         st.markdown("##### ATS Optimization Tips")
        
#         # ATS optimization tips
#         st.markdown("""
#         <div style='padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef;'>
#             <p style='color: #333333; margin: 5px 0;'><b>1. Use standard section headings</b> (Experience, Education, Skills)</p>
#             <p style='color: #333333; margin: 5px 0;'><b>2. Include exact skill keywords</b> from the job description</p>
#             <p style='color: #333333; margin: 5px 0;'><b>3. Avoid complex formatting,</b> tables, or graphics</p>
#             <p style='color: #333333; margin: 5px 0;'><b>4. Use a clean, single-column layout</b> for better parsing</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Prioritized action items
#     st.markdown("#### 🎯 Prioritized Action Items")
    
#     # Top skills to add (from priority list)
#     if total_missing > 0:
#         st.markdown("##### Top Skills to Add")
        
#         # Calculate priority skills
#         skill_priorities = calculate_skill_priorities(st_session_state.missing_skills, st_session_state.jd_text)
        
#         # Display top priority skills with action items
#         for i, (skill, count) in enumerate(skill_priorities[:3]):
#             st.markdown(f"""
#             <div style='padding: 12px; background-color: #f0f7ff; border-radius: 5px; margin: 10px 0; border: 1px solid #c0d6e9;'>
#                 <div style='font-size: 17px; color: #0a2540; margin-bottom: 5px;'><b>{i+1}. Add "{skill}" to your resume</b></div>
#                 <div style='font-size: 14px; color: #486581;'>
#                     This skill appears {count} times in the job description. Try to include it in your skills section and 
#                     demonstrate it through your work experience or projects.
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
    
#     # Resume language optimization
#     st.markdown("##### Resume Language Enhancement")
    
#     # Extract key phrases from job description for language alignment
#     common_job_phrases = ["experience with", "proficient in", "knowledge of", "ability to", "responsible for"]
#     jd_text_lower = st_session_state.jd_text.lower()
    
#     # Find matching phrases
#     found_phrases = []
#     for phrase in common_job_phrases:
#         if phrase in jd_text_lower:
#             found_phrases.append(phrase)
    
#     if found_phrases:
#         st.markdown("Align your resume language with these phrases from the job description:")
        
#         phrases_cols = st.columns(min(3, len(found_phrases)))
#         for i, phrase in enumerate(found_phrases[:3]):
#             with phrases_cols[i]:
#                 st.markdown(f"""
#                 <div style='text-align: center; padding: 10px; background-color: #e6f4ea; border-radius: 5px; border: 1px solid #b7dfb9;'>
#                     <span style='font-size: 16px; color: #0a2540;'><b>"{phrase}"</b></span>
#                 </div>
#                 """, unsafe_allow_html=True)
    
#     # Final tips box
#     st.markdown("#### 💡 Expert Resume Optimization Tips")
#     st.markdown("""
#     <div style='padding: 15px; background-color: #fff8e1; border-radius: 5px; border: 1px solid #ffe082;'>
#         <p style='color: #333333; margin: 5px 0;'><b>Tailor your resume for each application:</b> Customize your resume for each job rather than using a generic version</p>
#         <p style='color: #333333; margin: 5px 0;'><b>Quantify achievements:</b> Use numbers and percentages to showcase the impact of your work</p>
#         <p style='color: #333333; margin: 5px 0;'><b>Focus on recent experience:</b> Emphasize your most recent and relevant experience</p>
#         <p style='color: #333333; margin: 5px 0;'><b>Keep it concise:</b> Aim for 1-2 pages with clear, direct language</p>
#     </div>
#     """, unsafe_allow_html=True)

# def calculate_skill_priorities(missing_skills, jd_text):
#     """Calculate priority of missing skills based on frequency in job description"""
#     skill_priorities = {}
#     jd_lower = jd_text.lower()
    
#     for skill in missing_skills:
#         count = jd_lower.count(skill)
#         skill_priorities[skill] = count
    
#     # Sort by count (descending)
#     sorted_priorities = sorted(skill_priorities.items(), 
#                                key=lambda x: x[1], 
#                                reverse=True)
    
#     return sorted_priorities