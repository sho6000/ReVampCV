#!/usr/bin/env python3
"""
Script to install spaCy model with SSL certificate handling
"""

import subprocess
import sys
import os
import ssl
import urllib.request

def install_spacy_model():
    """Install spaCy English model with SSL certificate handling"""
    print("🔄 Installing spaCy English model...")
    
    try:
        # Method 1: Try direct download with SSL context
        print("📥 Attempting to download spaCy model...")
        
        # Create SSL context that ignores certificate verification
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Try to download the model using pip with SSL context
        if os.name == 'nt':  # Windows
            pip_cmd = "venv\\Scripts\\pip"
        else:  # Unix/Linux/macOS
            pip_cmd = "venv/bin/pip"
        
        # Method 1: Try with --trusted-host
        result = subprocess.run([
            pip_cmd, "install", 
            "--trusted-host", "pypi.org",
            "--trusted-host", "pypi.python.org",
            "--trusted-host", "files.pythonhosted.org",
            "--trusted-host", "github.com",
            "--trusted-host", "raw.githubusercontent.com",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ spaCy model installed successfully!")
            return True
        
        # Method 2: Try with python -m spacy download
        print("🔄 Trying alternative installation method...")
        if os.name == 'nt':  # Windows
            python_cmd = "venv\\Scripts\\python"
        else:  # Unix/Linux/macOS
            python_cmd = "venv/bin/python"
        
        # Set environment variables to handle SSL issues
        env = os.environ.copy()
        env['PYTHONHTTPSVERIFY'] = '0'
        
        result = subprocess.run([
            python_cmd, "-m", "spacy", "download", "en_core_web_sm"
        ], capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print("✅ spaCy model installed successfully!")
            return True
        
        # Method 3: Manual download and install
        print("🔄 Trying manual download method...")
        model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl"
        model_file = "en_core_web_sm-3.7.0-py3-none-any.whl"
        
        try:
            # Download the model file
            print(f"📥 Downloading {model_url}...")
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(model_url, model_file)
            
            # Install the downloaded file
            result = subprocess.run([
                pip_cmd, "install", model_file
            ], capture_output=True, text=True)
            
            # Clean up
            if os.path.exists(model_file):
                os.remove(model_file)
            
            if result.returncode == 0:
                print("✅ spaCy model installed successfully!")
                return True
                
        except Exception as e:
            print(f"❌ Manual download failed: {e}")
        
        # Method 4: Use conda if available
        print("🔄 Trying conda installation...")
        try:
            result = subprocess.run([
                "conda", "install", "-c", "conda-forge", "spacy-model-en_core_web_sm", "-y"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ spaCy model installed successfully via conda!")
                return True
        except FileNotFoundError:
            print("ℹ️ Conda not available, skipping conda method")
        
        print("❌ All installation methods failed")
        print("\n📋 Manual Installation Instructions:")
        print("1. Open a new terminal/command prompt")
        print("2. Activate your virtual environment:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   source venv/bin/activate")
        print("3. Run the following command:")
        print("   python -m spacy download en_core_web_sm")
        print("\nIf you still get SSL errors, try:")
        print("   pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org spacy")
        print("   python -m spacy download en_core_web_sm")
        
        return False
        
    except Exception as e:
        print(f"❌ Error installing spaCy model: {e}")
        return False

def verify_installation():
    """Verify that spaCy model is installed correctly"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Hello world!")
        print("✅ spaCy model verification successful!")
        return True
    except Exception as e:
        print(f"❌ spaCy model verification failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Installing spaCy English Model...")
    print("=" * 50)
    
    if install_spacy_model():
        print("\n🔍 Verifying installation...")
        if verify_installation():
            print("\n🎉 spaCy model installation completed successfully!")
        else:
            print("\n⚠️ Installation completed but verification failed.")
    else:
        print("\n❌ Installation failed. Please try manual installation.")

if __name__ == "__main__":
    main() 