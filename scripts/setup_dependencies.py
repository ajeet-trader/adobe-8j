"""
Setup script to install required dependencies
Run this first: python scripts/setup_dependencies.py
"""

import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Required packages for the PDF processing challenge
packages = [
    "streamlit",
    "PyPDF2",
    "pdfplumber", 
    "spacy",
    "transformers",
    "sentence-transformers",
    "scikit-learn",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "networkx",
    "plotly",
    "python-docx",
    "openpyxl",
    "nltk",
    "textstat",
    "wordcloud"
]

print("Installing required packages...")
for package in packages:
    try:
        install_package(package)
        print(f"✓ Installed {package}")
    except Exception as e:
        print(f"✗ Failed to install {package}: {e}")

print("\nDownloading spaCy model...")
try:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("✓ Downloaded spaCy English model")
except Exception as e:
    print(f"✗ Failed to download spaCy model: {e}")

print("\nSetup complete! You can now run the application with:")
print("streamlit run app.py")
