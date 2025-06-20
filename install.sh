#!/bin/bash

echo "🚀 Resume Analyzer Pro - Installation Script"
echo "================================================"

echo ""
echo "📋 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

python3 --version

echo ""
echo "🔄 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo ""
echo "📦 Installing Python dependencies..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🔄 Installing spaCy model..."
python install_spacy_model.py
if [ $? -ne 0 ]; then
    echo "⚠️ spaCy model installation failed, but application will still work with fallback"
fi

echo ""
echo "📝 Creating .env file..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Environment Variables for Resume Analyzer Pro

# OpenAI API Key (Optional - for enhanced AI features)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Debug Mode (Optional)
DEBUG=False

# Application Settings (Optional)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
EOF
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "================================================"
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your OpenAI API key (optional)"
echo "2. Run the application:"
echo "   streamlit run app.py"
echo "3. Open your browser to http://localhost:8501"
echo ""
echo "📚 For more information, check the README.md file"
echo "================================================" 