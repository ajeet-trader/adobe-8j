"""
Adobe India Hackathon 2025 - Challenge 1B: Adaptive Smart PDF Tool
Simplified version with proper multi-file upload support
"""

import streamlit as st
import fitz  # PyMuPDF
import json
import re
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Challenge 1B - Smart PDF Tool",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 1rem;
    }
    .collection-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF0000;
        margin: 1rem 0;
    }
    .relevance-high { background-color: #d4edda; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; }
    .relevance-medium { background-color: #fff3cd; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; }
    .relevance-low { background-color: #f8d7da; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

class SmartPDFAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def analyze_persona_and_task(self, persona: str, task: str) -> Dict[str, Any]:
        """Analyze persona and task to extract key characteristics"""
        
        # Extract meaningful keywords
        persona_words = self._extract_keywords(persona.lower())
        task_words = self._extract_keywords(task.lower())
        
        # Identify domain
        domain = self._identify_domain(persona, task)
        
        # Extract action verbs
        action_verbs = self._extract_action_verbs(task)
        
        return {
            "persona_keywords": persona_words,
            "task_keywords": task_words,
            "domain": domain,
            "action_verbs": action_verbs
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        words = word_tokenize(text)
        keywords = [
            word.lower() for word in words 
            if word.isalpha() and len(word) > 2 and word.lower() not in self.stop_words
        ]
        return list(set(keywords))
    
    def _identify_domain(self, persona: str, task: str) -> str:
        """Identify the domain based on persona and task"""
        text = f"{persona} {task}".lower()
        
        domains = {
            "travel": ["travel", "trip", "vacation", "hotel", "restaurant", "tourism"],
            "hr": ["hr", "human resources", "employee", "onboarding", "forms", "compliance"],
            "food": ["food", "recipe", "cooking", "menu", "catering", "restaurant"],
            "legal": ["legal", "law", "compliance", "regulation", "contract"],
            "finance": ["finance", "financial", "investment", "budget", "accounting"],
            "technology": ["software", "tech", "system", "development", "IT"],
            "healthcare": ["health", "medical", "patient", "clinical", "healthcare"],
            "business": ["business", "management", "strategy", "corporate"]
        }
        
        for domain, keywords in domains.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return "general"
    
    def _extract_action_verbs(self, task: str) -> List[str]:
        """Extract action verbs from task"""
        common_verbs = [
            "analyze", "create", "manage", "plan", "design", "develop", 
            "implement", "organize", "prepare", "build", "extract", 
            "identify", "find", "summarize", "review"
        ]
        
        task_lower = task.lower()
        found_verbs = [verb for verb in common_verbs if verb in task_lower]
        return found_verbs
    
    def calculate_relevance(self, text: str, analysis: Dict[str, Any]) -> float:
        """Calculate relevance score for text"""
        text_lower = text.lower()
        score = 0.0
        
        # Persona keywords (40% weight)
        persona_matches = sum(1 for word in analysis["persona_keywords"] if word in text_lower)
        if analysis["persona_keywords"]:
            score += (persona_matches / len(analysis["persona_keywords"])) * 0.4
        
        # Task keywords (30% weight)
        task_matches = sum(1 for word in analysis["task_keywords"] if word in text_lower)
        if analysis["task_keywords"]:
            score += (task_matches / len(analysis["task_keywords"])) * 0.3
        
        # Action verbs (20% weight)
        action_matches = sum(1 for verb in analysis["action_verbs"] if verb in text_lower)
        if analysis["action_verbs"]:
            score += (action_matches / len(analysis["action_verbs"])) * 0.2
        
        # Length bonus (10% weight)
        length_score = min(len(text) / 500, 1.0) * 0.1
        score += length_score
        
        return min(score, 1.0)
    
    def process_pdf_collection(self, uploaded_files: List, persona: str, task: str) -> Dict[str, Any]:
        """Process a collection of PDF files"""
        
        if len(uploaded_files) < 2:
            st.error("❌ Challenge 1B requires at least 2 PDF files. Please upload multiple documents.")
            return {}
        
        # Analyze persona and task
        analysis = self.analyze_persona_and_task(persona, task)
        
        st.info(f"🧠 Processing {len(uploaded_files)} documents for {analysis['domain']} domain...")
        
        all_sections = []
        all_content = []
        
        progress_bar = st.progress(0)
        
        # Process each PDF
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            sections, content = self._process_single_pdf(uploaded_file, analysis)
            all_sections.extend(sections)
            all_content.extend(content)
        
        # Rank by relevance
        all_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        all_content.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Assign ranks
        for i, section in enumerate(all_sections, 1):
            section["importance_rank"] = i
        
        progress_bar.progress(1.0)
        st.success("✅ Collection analysis complete!")
        
        return {
            "metadata": {
                "input_documents": [f.name for f in uploaded_files],
                "persona": persona,
                "job_to_be_done": task,
                "domain": analysis["domain"],
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_extracted": len(all_sections),
                "total_subsections": len(all_content),
                "persona_analysis": analysis
            },
            "extracted_sections": all_sections[:50],  # Top 50
            "subsection_analysis": all_content[:100]  # Top 100
        }
    
    def _process_single_pdf(self, uploaded_file, analysis: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """Process a single PDF file"""
        sections = []
        content = []
        
        try:
            # Read PDF
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        block_text = self._extract_text_from_block(block)
                        
                        if self._is_meaningful_text(block_text):
                            relevance_score = self.calculate_relevance(block_text, analysis)
                            
                            if relevance_score > 0.1:  # Minimum threshold
                                
                                # Check if it's a section title
                                if self._is_section_title(block, block_text) and relevance_score > 0.3:
                                    sections.append({
                                        "document": uploaded_file.name,
                                        "section_title": block_text[:100],
                                        "page_number": page_num + 1,
                                        "relevance_score": relevance_score
                                    })
                                
                                # Add to content analysis
                                content.append({
                                    "document": uploaded_file.name,
                                    "refined_text": block_text,
                                    "page_number": page_num + 1,
                                    "relevance_score": relevance_score,
                                    "relevance_category": self._categorize_relevance(relevance_score)
                                })
            
            doc.close()
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        
        return sections, content
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from a block"""
        text_parts = []
        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                line_text += span["text"]
            text_parts.append(line_text.strip())
        return " ".join(text_parts).strip()
    
    def _is_meaningful_text(self, text: str) -> bool:
        """Check if text is meaningful"""
        return len(text.strip()) > 10 and not re.match(r'^\d+$', text.strip())
    
    def _is_section_title(self, block: Dict, text: str) -> bool:
        """Check if text is likely a section title"""
        # Simple heuristic: short text with larger font
        avg_font_size = 0
        font_count = 0
        
        for line in block["lines"]:
            for span in line["spans"]:
                avg_font_size += span["size"]
                font_count += 1
        
        if font_count > 0:
            avg_font_size /= font_count
        
        return (
            len(text) < 100 and 
            avg_font_size > 12 and 
            len(text.split()) < 15
        )
    
    def _categorize_relevance(self, score: float) -> str:
        """Categorize relevance score"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

def main():
    """Main application"""
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SmartPDFAnalyzer()
    
    # Header
    st.markdown('<div class="main-header">🧠 Challenge 1B: Smart PDF Tool</div>', unsafe_allow_html=True)
    st.markdown("### Adaptive Analysis for ANY Persona and Task")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 How It Works")
        st.markdown("""
        1. **Define** your persona and task
        2. **Upload** multiple PDF files
        3. **Analyze** content across all documents
        4. **Get** ranked, relevant results
        """)
        
        st.markdown("## ✨ Features")
        st.markdown("""
        - 🧠 Adaptive to any persona
        - 📚 Multi-document analysis
        - 🎯 Relevance-based ranking
        - 📊 Interactive results
        """)
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Your Persona")
        persona = st.text_input(
            "Who are you?",
            placeholder="e.g., Data Scientist, Legal Advisor, Marketing Manager",
            help="Enter your professional role"
        )
    
    with col2:
        st.markdown("#### 🎯 Your Task")
        task = st.text_area(
            "What do you need to do?",
            placeholder="e.g., Extract key insights for quarterly report",
            height=100,
            help="Describe your specific goal"
        )
    
    if persona and task:
        # Show analysis preview
        analysis = st.session_state.analyzer.analyze_persona_and_task(persona, task)
        
        st.markdown('<div class="collection-info">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Analysis Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Domain:** {analysis['domain'].title()}")
        with col2:
            st.write(f"**Key Terms:** {len(analysis['persona_keywords']) + len(analysis['task_keywords'])}")
        with col3:
            st.write(f"**Actions:** {', '.join(analysis['action_verbs'][:3])}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # File upload - MULTIPLE FILES
        st.markdown("#### 📚 Upload Your Document Collection")
        uploaded_files = st.file_uploader(
            "Select multiple PDF files (minimum 2 required)",
            type="pdf",
            accept_multiple_files=True,  # This is the key!
            help="Upload 2 or more related PDF documents for analysis"
        )
        
        if uploaded_files:
            if len(uploaded_files) >= 2:
                st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
                
                # Show file details
                with st.expander("📋 Collection Details"):
                    for i, file in enumerate(uploaded_files, 1):
                        st.write(f"{i}. **{file.name}** ({file.size:,} bytes)")
                
                # Analyze button
                if st.button("🚀 Analyze Collection", type="primary", use_container_width=True):
                    result = st.session_state.analyzer.process_pdf_collection(
                        uploaded_files, persona, task
                    )
                    
                    if result:
                        st.session_state.analysis_result = result
                        st.rerun()
            
            else:
                st.warning(f"⚠️ Please upload at least 2 PDF files. You have uploaded {len(uploaded_files)} file(s).")
                st.info("💡 Challenge 1B requires analyzing multiple documents together to find connections and insights.")
    
    # Display results
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        metadata = result.get('metadata', {})
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", len(metadata.get('input_documents', [])))
        with col2:
            st.metric("🏆 Sections", metadata.get('total_sections_extracted', 0))
        with col3:
            st.metric("📝 Content Items", metadata.get('total_subsections', 0))
        with col4:
            st.metric("🎯 Domain", metadata.get('domain', 'general').title())
        
        # Tabs for results
        tab1, tab2, tab3 = st.tabs(["🏆 Top Sections", "📝 Content Analysis", "💾 Export"])
        
        with tab1:
            sections = result.get('extracted_sections', [])
            if sections:
                st.markdown(f"### Top {len(sections)} Most Relevant Sections")
                
                # Create DataFrame
                df_data = []
                for section in sections:
                    df_data.append({
                        'Rank': section.get('importance_rank', 0),
                        'Document': section.get('document', ''),
                        'Section': section.get('section_title', ''),
                        'Page': section.get('page_number', 0),
                        'Relevance': f"{section.get('relevance_score', 0):.3f}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Visualization
                doc_counts = df['Document'].value_counts()
                fig = px.bar(
                    x=doc_counts.index,
                    y=doc_counts.values,
                    title="Sections per Document",
                    labels={'x': 'Document', 'y': 'Number of Sections'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            content = result.get('subsection_analysis', [])
            if content:
                st.markdown(f"### Content Analysis ({len(content)} items)")
                
                # Filter by relevance
                relevance_filter = st.selectbox(
                    "Filter by relevance:",
                    ["All", "High", "Medium", "Low"]
                )
                
                filtered_content = content
                if relevance_filter != "All":
                    filtered_content = [
                        item for item in content 
                        if item.get('relevance_category', '').lower() == relevance_filter.lower()
                    ]
                
                # Group by document
                doc_content = defaultdict(list)
                for item in filtered_content:
                    doc_content[item['document']].append(item)
                
                # Display content by document
                for doc_name, items in doc_content.items():
                    with st.expander(f"📄 {doc_name} ({len(items)} items)"):
                        for i, item in enumerate(items[:5], 1):  # Show top 5 per doc
                            category = item.get('relevance_category', 'low')
                            score = item.get('relevance_score', 0)
                            
                            if category == 'high':
                                st.markdown('<div class="relevance-high">', unsafe_allow_html=True)
                            elif category == 'medium':
                                st.markdown('<div class="relevance-medium">', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="relevance-low">', unsafe_allow_html=True)
                            
                            st.markdown(f"**Item {i} (Page {item['page_number']}) - {category.title()} Relevance**")
                            st.progress(score, text=f"Score: {score:.3f}")
                            
                            text = item['refined_text']
                            display_text = text[:300] + "..." if len(text) > 300 else text
                            st.write(display_text)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export JSON
                if st.button("📄 Export Complete Analysis (JSON)"):
                    json_data = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        "analysis_results.json",
                        "application/json"
                    )
            
            with col2:
                # Export CSV
                sections = result.get('extracted_sections', [])
                if sections:
                    if st.button("📊 Export Sections (CSV)"):
                        df_data = []
                        for section in sections:
                            df_data.append({
                                'rank': section.get('importance_rank', 0),
                                'document': section.get('document', ''),
                                'section_title': section.get('section_title', ''),
                                'page_number': section.get('page_number', 0),
                                'relevance_score': section.get('relevance_score', 0)
                            })
                        
                        df = pd.DataFrame(df_data)
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv_data,
                            "sections.csv",
                            "text/csv"
                        )
        
        # Reset button
        if st.button("🔄 Start New Analysis", type="secondary"):
            if 'analysis_result' in st.session_state:
                del st.session_state.analysis_result
            st.rerun()

if __name__ == "__main__":
    main()
