#!/bin/bash

# Adobe India Hackathon 2025 - Challenge 1B Runner Script

echo "🚀 Starting Challenge 1B - Multi-Collection PDF Analysis"
echo "======================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements_challenge1b.txt

# Create directory structure if it doesn't exist
echo "📁 Setting up directory structure..."
mkdir -p "Collection 1/PDFs"
mkdir -p "Collection 2/PDFs"
mkdir -p "Collection 3/PDFs"

# Copy sample input files if they don't exist
if [ ! -f "Collection 1/challenge1b_input.json" ]; then
    cp sample_input_collection1.json "Collection 1/challenge1b_input.json"
fi

if [ ! -f "Collection 2/challenge1b_input.json" ]; then
    cp sample_input_collection2.json "Collection 2/challenge1b_input.json"
fi

if [ ! -f "Collection 3/challenge1b_input.json" ]; then
    cp sample_input_collection3.json "Collection 3/challenge1b_input.json"
fi

# Run the Streamlit application
echo "🔄 Starting Streamlit application..."
streamlit run challenge1b_app.py

echo "🎉 Challenge 1B application started!"
echo "Open your browser and go to the URL shown above."
