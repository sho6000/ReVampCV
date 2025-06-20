# 📄 Resume Analyzer Pro

An AI-powered resume analysis and optimization tool that helps job seekers improve their resumes by comparing them against job descriptions and providing actionable feedback.

## 🚀 Features

- **📊 Smart Analysis**: AI-powered resume analysis using advanced NLP techniques
- **🎯 Skills Matching**: Compare your skills with job requirements
- **💡 Recommendations**: Get actionable improvement suggestions
- **✨ Auto-Improvement**: Generate enhanced resume versions automatically
- **📋 Side-by-Side Comparison**: View original vs improved resume
- **📥 Download Options**: Download improved resumes in multiple formats
- **🎨 Modern UI**: Beautiful and intuitive user interface
- **📄 Multi-Format Support**: Upload PDF, DOCX, and TXT files

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI/ML**: OpenAI GPT-3.5, spaCy, LangChain
- **Document Processing**: PyPDF2, python-docx
- **Data Visualization**: Plotly
- **NLP**: spaCy, textstat

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional, for enhanced AI features)
- Internet connection

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Resume-Analyzer
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables (optional)**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:8501`

3. **Upload files**
   - Upload your resume (PDF, DOCX, or TXT)
   - Upload the job description (PDF, DOCX, or TXT)

4. **Analyze**
   Click the "Analyze Resume" button to get comprehensive feedback

5. **Review results**
   - View overall match score
   - Check skills analysis
   - Read recommendations
   - Generate improved resume

## 📊 Features in Detail

### 🔍 Smart Analysis
- **Overall Score**: Percentage match between resume and job description
- **Skills Analysis**: Identifies matched and missing skills
- **Experience Evaluation**: Assesses experience relevance
- **Education Assessment**: Evaluates educational background

### 💡 Recommendations
- **Actionable Suggestions**: Specific improvements for your resume
- **Skills Enhancement**: Add missing skills and keywords
- **Content Optimization**: Improve descriptions and achievements
- **Format Suggestions**: Better structure and presentation

### ✨ Auto-Improvement
- **AI-Generated Content**: Enhanced resume sections
- **Keyword Optimization**: Better alignment with job requirements
- **Professional Language**: Improved writing style
- **Achievement Quantification**: Add specific metrics and results

### 📋 Comparison View
- **Side-by-Side Display**: Original vs improved resume
- **Highlighted Changes**: Clear indication of improvements
- **Download Options**: Save enhanced versions

## 🎨 UI/UX Features

### Modern Design
- **Gradient Backgrounds**: Beautiful color schemes
- **Card-based Layout**: Clean and organized presentation
- **Interactive Elements**: Hover effects and animations
- **Responsive Design**: Works on all screen sizes

### User Experience
- **Intuitive Navigation**: Easy-to-use interface
- **Progress Indicators**: Clear workflow guidance
- **Error Handling**: Helpful error messages
- **Success Feedback**: Positive reinforcement

### Visual Elements
- **Gauge Charts**: Score visualization
- **Skill Tags**: Color-coded skill indicators
- **Progress Bars**: Completion tracking
- **Icons**: Meaningful visual cues

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for enhanced AI features
- `DEBUG`: Set to `True` for debug mode

### Customization
- Modify `skills_database` in `resume_analyzer.py` to add custom skills
- Update CSS styles in `app.py` for different themes
- Add new file formats in `resume_analyzer.py`

## 📁 Project Structure

```
Resume-Analyzer/
├── app.py                 # Main Streamlit application
├── resume_analyzer.py     # Core analysis logic
├── ui_components.py       # UI components and helpers
├── sample_data.py         # Sample data for demonstration
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .env                  # Environment variables (create this)
```

## 🧪 Testing

### Sample Data
Use the built-in sample data to test the application:
1. Click "Try Sample Analysis" in the sidebar
2. Upload sample files or use demo mode
3. Review the analysis results

### Manual Testing
1. Upload different file formats
2. Test with various resume types
3. Verify analysis accuracy
4. Check download functionality

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment
1. **Heroku**
   - Create `Procfile`: `web: streamlit run app.py`
   - Add `setup.sh` for dependencies
   - Deploy via Heroku CLI

2. **Streamlit Cloud**
   - Connect your GitHub repository
   - Deploy directly from Streamlit Cloud

3. **AWS/GCP/Azure**
   - Use containerization with Docker
   - Deploy to cloud platforms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-3.5 API
- Streamlit for the web framework
- spaCy for NLP capabilities
- The open-source community for various libraries

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

## 🔄 Updates

### Version 1.0.0
- Initial release
- Basic resume analysis
- AI-powered improvements
- Modern UI/UX design

### Future Enhancements
- Multi-language support
- Advanced AI models
- Integration with job boards
- Resume templates
- ATS optimization

---

**Made with ❤️ for job seekers worldwide** 