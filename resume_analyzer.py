import re
import spacy
import textstat
from typing import Dict, List, Tuple
import openai
from docx import Document
import PyPDF2
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ResumeAnalyzer:
    def __init__(self):
        """Initialize the Resume Analyzer with necessary components"""
        # Try to load spaCy model, with fallback if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except OSError:
            print("⚠️ spaCy model not found. Using fallback text processing.")
            self.nlp = None
            self.spacy_available = False
        
        # Initialize OpenAI (you'll need to set OPENAI_API_KEY in your .env file)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Common skills database
        self.skills_database = {
            'technical': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask',
                'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
                'machine learning', 'ai', 'data science', 'pandas', 'numpy', 'tensorflow',
                'pytorch', 'scikit-learn', 'tableau', 'power bi', 'excel', 'r'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'critical thinking', 'time management', 'project management',
                'collaboration', 'adaptability', 'creativity', 'analytical thinking'
            ],
            'business': [
                'agile', 'scrum', 'kanban', 'lean', 'six sigma', 'budgeting',
                'strategic planning', 'market analysis', 'customer service',
                'sales', 'marketing', 'business development'
            ]
        }
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded file (PDF, DOCX, TXT)"""
        try:
            if uploaded_file.type == "application/pdf":
                return self._extract_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self._extract_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                return uploaded_file.getvalue().decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")
        except Exception as e:
            raise Exception(f"Error extracting text from file: {str(e)}")
    
    def _extract_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def _extract_from_docx(self, uploaded_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using NLP and pattern matching"""
        text_lower = text.lower()
        found_skills = []
        
        # Check for skills in database
        for category, skills in self.skills_database.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)
        
        # Use NLP to find additional skills if available
        if self.spacy_available and self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract noun phrases that might be skills
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:  # Skills are usually 1-3 words
                        skill = chunk.text.lower().strip()
                        if skill not in found_skills and len(skill) > 2:
                            found_skills.append(skill)
            except Exception as e:
                print(f"Warning: spaCy processing failed: {e}")
        
        # Fallback: Use regex patterns to find skills
        if not self.spacy_available:
            # Look for common skill patterns
            skill_patterns = [
                r'\b(?:proficient in|experience with|skilled in|knowledge of)\s+([a-zA-Z\s\+]+)',
                r'\b([a-zA-Z\s\+]+)\s+(?:development|programming|administration|management)',
                r'\b(?:using|with)\s+([a-zA-Z\s\+]+)',
            ]
            
            for pattern in skill_patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    if isinstance(match, tuple):
                        skill = match[0].strip()
                    else:
                        skill = match.strip()
                    if skill and skill not in found_skills and len(skill) > 2:
                        found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    def extract_experience_info(self, text: str) -> Dict:
        """Extract experience information from resume"""
        experience_info = {
            'years_of_experience': 0,
            'job_titles': [],
            'companies': []
        }
        
        # Look for years of experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*of?\s*experience',
            r'experience:\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*the\s*field'
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                experience_info['years_of_experience'] = max(
                    experience_info['years_of_experience'],
                    max([int(match) for match in matches])
                )
        
        # Extract job titles
        job_title_patterns = [
            r'(senior|junior|lead|principal|staff)?\s*(software engineer|developer|programmer|analyst|manager|director|consultant)',
            r'(data scientist|machine learning engineer|devops engineer|product manager)',
            r'(frontend|backend|full stack|mobile|web)\s*(developer|engineer)'
        ]
        
        for pattern in job_title_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    job_title = ' '.join(match).strip()
                else:
                    job_title = match.strip()
                if job_title and job_title not in experience_info['job_titles']:
                    experience_info['job_titles'].append(job_title)
        
        return experience_info
    
    def calculate_skills_match(self, resume_skills: List[str], job_skills: List[str]) -> Dict:
        """Calculate skills match between resume and job description"""
        if not job_skills:
            return {'match_percentage': 0, 'matched_skills': [], 'missing_skills': []}
        
        matched_skills = []
        missing_skills = []
        
        for job_skill in job_skills:
            # Check for exact matches and partial matches
            found = False
            for resume_skill in resume_skills:
                if (job_skill.lower() in resume_skill.lower() or 
                    resume_skill.lower() in job_skill.lower()):
                    matched_skills.append(job_skill)
                    found = True
                    break
            
            if not found:
                missing_skills.append(job_skill)
        
        match_percentage = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0
        
        return {
            'match_percentage': match_percentage,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills
        }
    
    def analyze_resume_with_ai(self, resume_text: str, job_description: str) -> Dict:
        """Use AI to analyze resume against job description"""
        try:
            if not openai.api_key:
                # Fallback to rule-based analysis if no API key
                return self._rule_based_analysis(resume_text, job_description)
            
            prompt = f"""
            Analyze the following resume against the job description and provide a comprehensive analysis.
            
            Job Description:
            {job_description}
            
            Resume:
            {resume_text}
            
            Please provide analysis in the following JSON format:
            {{
                "overall_score": <score from 0-100>,
                "skills_analysis": {{
                    "matched_skills": ["skill1", "skill2"],
                    "missing_skills": ["skill1", "skill2"],
                    "skills_score": <score from 0-100>
                }},
                "experience_analysis": {{
                    "relevance_score": <score from 0-100>,
                    "years_match": <boolean>,
                    "role_alignment": <score from 0-100>
                }},
                "education_analysis": {{
                    "education_score": <score from 0-100>,
                    "degree_relevance": <score from 0-100>
                }},
                "recommendations": [
                    "specific recommendation 1",
                    "specific recommendation 2"
                ],
                "suggested_improvements": [
                    "specific improvement 1",
                    "specific improvement 2"
                ]
            }}
            
            Focus on:
            1. Skills alignment
            2. Experience relevance
            3. Education match
            4. Specific actionable recommendations
            5. Areas for improvement
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume analyzer and career advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            # Extract JSON from the response
            import json
            try:
                # Find JSON in the response
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                json_str = analysis_text[start_idx:end_idx]
                return json.loads(json_str)
            except:
                # Fallback to rule-based analysis
                return self._rule_based_analysis(resume_text, job_description)
                
        except Exception as e:
            # Fallback to rule-based analysis
            return self._rule_based_analysis(resume_text, job_description)
    
    def _rule_based_analysis(self, resume_text: str, job_description: str) -> Dict:
        """Fallback rule-based analysis when AI is not available"""
        # Extract skills
        resume_skills = self.extract_skills_from_text(resume_text)
        job_skills = self.extract_skills_from_text(job_description)
        
        # Calculate skills match
        skills_analysis = self.calculate_skills_match(resume_skills, job_skills)
        
        # Extract experience info
        resume_experience = self.extract_experience_info(resume_text)
        job_experience = self.extract_experience_info(job_description)
        
        # Calculate experience score
        experience_score = 0
        if job_experience['years_of_experience'] > 0:
            if resume_experience['years_of_experience'] >= job_experience['years_of_experience']:
                experience_score = 100
            else:
                experience_score = (resume_experience['years_of_experience'] / job_experience['years_of_experience']) * 100
        
        # Calculate overall score
        overall_score = (skills_analysis['match_percentage'] * 0.6 + experience_score * 0.4)
        
        # Generate recommendations
        recommendations = []
        if skills_analysis['missing_skills']:
            recommendations.append(f"Add missing skills: {', '.join(skills_analysis['missing_skills'][:3])}")
        
        if experience_score < 80:
            recommendations.append("Highlight relevant experience more prominently")
        
        if overall_score < 70:
            recommendations.append("Consider adding more relevant projects and achievements")
        
        return {
            "overall_score": min(overall_score, 100),
            "matched_skills": skills_analysis['matched_skills'],
            "missing_skills": skills_analysis['missing_skills'],
            "experience_score": experience_score,
            "education_score": 75,  # Default score
            "recommendations": recommendations,
            "suggested_improvements": recommendations
        }
    
    def analyze_resume(self, resume_text: str, job_description: str) -> Dict:
        """Main method to analyze resume against job description"""
        # Use AI analysis with fallback
        analysis = self.analyze_resume_with_ai(resume_text, job_description)
        
        # Ensure all required fields are present
        required_fields = [
            'overall_score', 'matched_skills', 'missing_skills', 
            'experience_score', 'education_score', 'recommendations', 
            'suggested_improvements'
        ]
        
        for field in required_fields:
            if field not in analysis:
                analysis[field] = [] if field in ['matched_skills', 'missing_skills', 'recommendations', 'suggested_improvements'] else 0
        
        return analysis
    
    def generate_improved_resume(self, original_resume: str, job_description: str, analysis_results: Dict) -> str:
        """Generate an improved version of the resume"""
        try:
            if not openai.api_key:
                return self._generate_basic_improvements(original_resume, job_description, analysis_results)
            
            prompt = f"""
            Based on the analysis results, improve the following resume to better match the job description.
            
            Job Description:
            {job_description}
            
            Original Resume:
            {original_resume}
            
            Analysis Results:
            {analysis_results}
            
            Please provide an improved version of the resume that:
            1. Incorporates missing skills and keywords from the job description
            2. Highlights relevant experience more prominently
            3. Uses action verbs and quantifiable achievements
            4. Maintains the original structure and format
            5. Adds relevant sections if missing
            
            Return only the improved resume text without any explanations.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume writer and career advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return self._generate_basic_improvements(original_resume, job_description, analysis_results)
    
    def _generate_basic_improvements(self, original_resume: str, job_description: str, analysis_results: Dict) -> str:
        """Generate basic improvements when AI is not available"""
        improved_resume = original_resume
        
        # Add missing skills to skills section
        missing_skills = analysis_results.get('missing_skills', [])
        if missing_skills:
            skills_to_add = ', '.join(missing_skills[:5])  # Add top 5 missing skills
            improved_resume += f"\n\nAdditional Skills: {skills_to_add}"
        
        # Add recommendations as comments
        recommendations = analysis_results.get('recommendations', [])
        if recommendations:
            improved_resume += "\n\nSuggested Improvements:"
            for rec in recommendations:
                improved_resume += f"\n- {rec}"
        
        return improved_resume 