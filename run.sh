#!/bin/bash

echo "🚀 Starting Challenge 1B Smart PDF Tool"
echo "======================================"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading language data..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ Language data ready')
except:
    print('⚠️ Language data download failed - basic functionality will still work')
"

# Run the app
echo "🚀 Starting application..."
streamlit run app.py

echo "🎉 Application started! Open the URL shown above."
