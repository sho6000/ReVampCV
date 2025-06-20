#!/usr/bin/env python3
"""
Setup script for Resume Analyzer Pro
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    if not os.path.exists("venv"):
        return run_command("python -m venv venv", "Creating virtual environment")
    else:
        print("✅ Virtual environment already exists")
        return True

def install_dependencies():
    """Install required dependencies"""
    # Determine the correct pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")

def download_spacy_model():
    """Download spaCy English model"""
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    return run_command(f"{python_cmd} -m spacy download en_core_web_sm", "Downloading spaCy model")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# Environment Variables for Resume Analyzer Pro\n\n")
            f.write("# OpenAI API Key (Optional - for enhanced AI features)\n")
            f.write("# Get your API key from: https://platform.openai.com/api-keys\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Debug Mode (Optional)\n")
            f.write("DEBUG=False\n\n")
            f.write("# Application Settings (Optional)\n")
            f.write("STREAMLIT_SERVER_PORT=8501\n")
            f.write("STREAMLIT_SERVER_ADDRESS=localhost\n")
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up Resume Analyzer Pro...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key (optional)")
    print("2. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    print("3. Run the application:")
    print("   streamlit run app.py")
    print("4. Open your browser to http://localhost:8501")
    print("\n📚 For more information, check the README.md file")

if __name__ == "__main__":
    main() 