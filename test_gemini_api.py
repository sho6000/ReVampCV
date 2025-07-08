# Create a new file: test_gemini_api.py
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        # Remove quotes if they exist
        api_key = api_key.strip('"').strip("'")
        genai.configure(api_key=api_key)
        st.success(f"✅ API Key loaded: {api_key[:20]}...")
    else:
        st.error("❌ API Key not found in environment variables")
except Exception as e:
    st.error(f"❌ Error configuring API: {str(e)}")

st.title("🤖 Gemini API Test")
st.write("This app tests your Google Gemini API key")

# Display API key status
st.subheader("API Key Status")
if api_key:
    st.write(f"API Key: {api_key[:20]}...")
else:
    st.write("No API key found")

# Test input
st.subheader("Test the API")
test_prompt = st.text_area(
    "Enter a test prompt:",
    value="Write a short professional summary for a software engineer with 3 years of experience in Python and React.",
    height=100
)

if st.button("Test API", use_container_width=True):
    if not api_key:
        st.error("❌ Please configure your API key first")
    else:
        with st.spinner("Testing API connection..."):
            try:
                # Initialize model
                model = genai.GenerativeModel('gemini-pro')
                
                # Generate response
                response = model.generate_content(test_prompt)
                
                # Display results
                st.success("✅ API is working!")
                st.subheader("API Response:")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")


# Environment check
st.subheader("Environment Check")
st.write("Checking required packages:")

try:
    import google.generativeai
    st.write("✅ google-generativeai installed")
except ImportError:
    st.write("❌ google-generativeai not installed")
    st.code("pip install google-generativeai")

try:
    from dotenv import load_dotenv
    st.write("✅ python-dotenv installed")
except ImportError:
    st.write("❌ python-dotenv not installed")
    st.code("pip install python-dotenv")