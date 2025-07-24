#!/bin/bash

# Adobe India Hackathon 2025 - Adaptive Challenge 1B Runner Script

echo "🧠 Starting Adaptive Smart PDF Tool - Challenge 1B"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements_adaptive.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️ NLTK download warning: {e}')
"

# Download spaCy model (optional)
echo "🧠 Downloading spaCy model (optional)..."
python3 -m spacy download en_core_web_sm || echo "⚠️ spaCy model download failed - some features may be limited"

# Run the Streamlit application
echo "🚀 Starting Adaptive Smart PDF Tool..."
streamlit run adaptive_challenge1b_app.py

echo "🎉 Adaptive Challenge 1B application started!"
echo "Open your browser and go to the URL shown above."
