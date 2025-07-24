"""
Core PDF processing module for extracting text, structure, and insights
"""

import PyPDF2
import pdfplumber
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple, Any
import nltk
from nltk.tokenize import sent_tokenize
import textstat

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor with NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features may be limited.")
            self.nlp = None
        
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def extract_text_from_pdf(self, pdf_file) -> Dict[str, Any]:
        """Extract text and metadata from PDF file"""
        text_content = []
        metadata = {}
        
        try:
            # Use pdfplumber for better text extraction
            with pdfplumber.open(pdf_file) as pdf:
                metadata = {
                    'total_pages': len(pdf.pages),
                    'title': getattr(pdf.metadata, 'title', 'Unknown'),
                    'author': getattr(pdf.metadata, 'author', 'Unknown'),
                    'subject': getattr(pdf.metadata, 'subject', 'Unknown')
                }
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append({
                            'page': page_num + 1,
                            'text': page_text,
                            'word_count': len(page_text.split())
                        })
        
        except Exception as e:
            print(f"Error extracting text: {e}")
            return {'text_content': [], 'metadata': {}}
        
        return {
            'text_content': text_content,
            'metadata': metadata,
            'full_text': ' '.join([page['text'] for page in text_content])
        }
    
    def extract_structure(self, text_content: List[Dict]) -> Dict[str, Any]:
        """Extract document structure including headings, sections, and hierarchy"""
        structure = {
            'headings': [],
            'sections': [],
            'outline': [],
            'toc': []
        }
        
        # Patterns for detecting headings and structure
        heading_patterns = [
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS headings
            r'^\d+\.\s+[A-Z].*',     # Numbered headings (1. Title)
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?$',  # Title Case headings
            r'^\d+\.\d+\s+.*',       # Sub-numbered headings (1.1 Title)
            r'^Chapter\s+\d+.*',     # Chapter headings
            r'^Section\s+\d+.*',     # Section headings
        ]
        
        current_section = None
        section_content = []
        
        for page_data in text_content:
            lines = page_data['text'].split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line matches heading patterns
                is_heading = False
                heading_level = 0
                
                for i, pattern in enumerate(heading_patterns):
                    if re.match(pattern, line):
                        is_heading = True
                        heading_level = i + 1
                        break
                
                if is_heading:
                    # Save previous section
                    if current_section and section_content:
                        structure['sections'].append({
                            'title': current_section,
                            'content': ' '.join(section_content),
                            'page': page_data['page'],
                            'word_count': len(' '.join(section_content).split())
                        })
                    
                    # Start new section
                    current_section = line
                    section_content = []
                    
                    structure['headings'].append({
                        'title': line,
                        'level': heading_level,
                        'page': page_data['page']
                    })
                    
                    structure['toc'].append({
                        'title': line,
                        'page': page_data['page'],
                        'level': heading_level
                    })
                else:
                    section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            structure['sections'].append({
                'title': current_section,
                'content': ' '.join(section_content),
                'page': text_content[-1]['page'] if text_content else 1,
                'word_count': len(' '.join(section_content).split())
            })
        
        return structure
    
    def analyze_content_relationships(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analyze relationships between different sections using semantic similarity"""
        if not self.sentence_model or not sections:
            return {'similarity_matrix': [], 'clusters': [], 'related_sections': []}
        
        # Extract section texts
        section_texts = [section['content'] for section in sections]
        section_titles = [section['title'] for section in sections]
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode(section_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find clusters of related content
            n_clusters = min(5, len(sections))  # Max 5 clusters
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
            else:
                clusters = [0] * len(sections)
            
            # Find most related sections for each section
            related_sections = []
            for i, section in enumerate(sections):
                similarities = similarity_matrix[i]
                # Get top 3 most similar sections (excluding self)
                similar_indices = np.argsort(similarities)[::-1][1:4]
                
                related = []
                for idx in similar_indices:
                    if similarities[idx] > 0.3:  # Threshold for relevance
                        related.append({
                            'title': section_titles[idx],
                            'similarity': float(similarities[idx]),
                            'index': int(idx)
                        })
                
                related_sections.append({
                    'section': section_titles[i],
                    'related': related
                })
            
            return {
                'similarity_matrix': similarity_matrix.tolist(),
                'clusters': clusters.tolist(),
                'related_sections': related_sections,
                'section_titles': section_titles
            }
        
        except Exception as e:
            print(f"Error in relationship analysis: {e}")
            return {'similarity_matrix': [], 'clusters': [], 'related_sections': []}
    
    def extract_key_insights(self, full_text: str, sections: List[Dict]) -> Dict[str, Any]:
        """Extract key insights, entities, and statistics from the document"""
        insights = {
            'key_entities': [],
            'statistics': {},
            'key_phrases': [],
            'readability': {},
            'summary_points': []
        }
        
        try:
            # Basic statistics
            insights['statistics'] = {
                'total_words': len(full_text.split()),
                'total_characters': len(full_text),
                'total_sentences': len(sent_tokenize(full_text)),
                'total_sections': len(sections),
                'avg_words_per_section': np.mean([s['word_count'] for s in sections]) if sections else 0
            }
            
            # Readability analysis
            insights['readability'] = {
                'flesch_reading_ease': textstat.flesch_reading_ease(full_text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(full_text),
                'automated_readability_index': textstat.automated_readability_index(full_text),
                'reading_time_minutes': len(full_text.split()) / 200  # Average reading speed
            }
            
            # Extract entities using spaCy if available
            if self.nlp:
                doc = self.nlp(full_text[:1000000])  # Limit text length for processing
                
                entities = {}
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    if ent.text not in entities[ent.label_]:
                        entities[ent.label_].append(ent.text)
                
                # Get top entities by category
                for label, ents in entities.items():
                    if len(ents) > 0:
                        insights['key_entities'].append({
                            'category': label,
                            'entities': ents[:10]  # Top 10 entities per category
                        })
            
            # Extract key phrases (simple approach)
            sentences = sent_tokenize(full_text)
            if sentences:
                # Get sentences with high information content
                key_sentences = []
                for sentence in sentences[:50]:  # Analyze first 50 sentences
                    if len(sentence.split()) > 10 and len(sentence.split()) < 30:
                        key_sentences.append(sentence.strip())
                
                insights['summary_points'] = key_sentences[:5]  # Top 5 key sentences
        
        except Exception as e:
            print(f"Error extracting insights: {e}")
        
        return insights
    
    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Main method to process PDF and return comprehensive analysis"""
        print("Extracting text from PDF...")
        extraction_result = self.extract_text_from_pdf(pdf_file)
        
        if not extraction_result['text_content']:
            return {'error': 'Could not extract text from PDF'}
        
        print("Analyzing document structure...")
        structure = self.extract_structure(extraction_result['text_content'])
        
        print("Analyzing content relationships...")
        relationships = self.analyze_content_relationships(structure['sections'])
        
        print("Extracting key insights...")
        insights = self.extract_key_insights(
            extraction_result['full_text'], 
            structure['sections']
        )
        
        return {
            'metadata': extraction_result['metadata'],
            'structure': structure,
            'relationships': relationships,
            'insights': insights,
            'text_content': extraction_result['text_content']
        }
