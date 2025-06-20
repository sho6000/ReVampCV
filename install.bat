@echo off
echo 🚀 Resume Analyzer Pro - Installation Script
echo ================================================

echo.
echo 📋 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo 🔄 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo ✅ Virtual environment already exists
)

echo.
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo 📦 Installing Python dependencies...
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo 🔄 Installing spaCy model...
python install_spacy_model.py
if %errorlevel% neq 0 (
    echo ⚠️ spaCy model installation failed, but application will still work with fallback
)

echo.
echo 📝 Creating .env file...
if not exist ".env" (
    echo # Environment Variables for Resume Analyzer Pro > .env
    echo. >> .env
    echo # OpenAI API Key (Optional - for enhanced AI features) >> .env
    echo # Get your API key from: https://platform.openai.com/api-keys >> .env
    echo OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo. >> .env
    echo # Debug Mode (Optional) >> .env
    echo DEBUG=False >> .env
    echo ✅ Created .env file
) else (
    echo ✅ .env file already exists
)

echo.
echo ================================================
echo 🎉 Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Edit .env file and add your OpenAI API key (optional)
echo 2. Run the application:
echo    streamlit run app.py
echo 3. Open your browser to http://localhost:8501
echo.
echo 📚 For more information, check the README.md file
echo ================================================
pause 