"""
Adobe India Hackathon 2025 - Challenge 1B: Smart PDF Tool
Fixed version with improved text extraction and content analysis
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
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

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
    .relevance-high { 
        background-color: #d4edda; 
        padding: 1rem; 
        border-radius: 8px; 
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .relevance-medium { 
        background-color: #fff3cd; 
        padding: 1rem; 
        border-radius: 8px; 
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    .relevance-low { 
        background-color: #f8d7da; 
        padding: 1rem; 
        border-radius: 8px; 
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
    }
    .debug-info {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SmartPDFAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def analyze_persona_and_task(self, persona: str, task: str) -> Dict[str, Any]:
        """Analyze persona and task to extract key characteristics"""
        
        # Extract meaningful keywords
        persona_words = self._extract_keywords(persona.lower())
        task_words = self._extract_keywords(task.lower())
        
        # Identify domain
        domain = self._identify_domain(persona, task)
        
        # Extract action verbs
        action_verbs = self._extract_action_verbs(task)
        
        # Add domain-specific keywords
        domain_keywords = self._get_domain_keywords(domain)
        
        return {
            "persona_keywords": persona_words,
            "task_keywords": task_words,
            "domain": domain,
            "action_verbs": action_verbs,
            "domain_keywords": domain_keywords
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Simple word extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in self.stop_words]
        return list(set(keywords))
    
    def _identify_domain(self, persona: str, task: str) -> str:
        """Identify the domain based on persona and task"""
        text = f"{persona} {task}".lower()
        
        domains = {
            "travel": ["travel", "trip", "vacation", "hotel", "restaurant", "tourism", "destination", "booking"],
            "hr": ["hr", "human resources", "employee", "onboarding", "forms", "compliance", "training"],
            "food": ["food", "recipe", "cooking", "menu", "catering", "restaurant", "chef", "kitchen", "ingredient"],
            "legal": ["legal", "law", "compliance", "regulation", "contract", "attorney"],
            "finance": ["finance", "financial", "investment", "budget", "accounting", "money"],
            "technology": ["software", "tech", "system", "development", "IT", "programming"],
            "healthcare": ["health", "medical", "patient", "clinical", "healthcare", "doctor"],
            "business": ["business", "management", "strategy", "corporate", "company"]
        }
        
        for domain, keywords in domains.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return "general"
    
    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get additional keywords for specific domains"""
        domain_keywords = {
            "travel": ["itinerary", "accommodation", "attraction", "transport", "guide", "culture", "local"],
            "hr": ["policy", "procedure", "workflow", "document", "signature", "approval", "process"],
            "food": ["vegetarian", "gluten-free", "buffet", "serving", "preparation", "dietary", "nutrition"],
            "legal": ["clause", "agreement", "liability", "terms", "conditions", "rights"],
            "finance": ["profit", "loss", "revenue", "cost", "analysis", "report", "statement"],
            "technology": ["feature", "function", "interface", "user", "data", "security"],
            "healthcare": ["treatment", "diagnosis", "therapy", "care", "wellness", "prevention"],
            "business": ["objective", "goal", "target", "performance", "efficiency", "productivity"]
        }
        return domain_keywords.get(domain, [])
    
    def _extract_action_verbs(self, task: str) -> List[str]:
        """Extract action verbs from task"""
        common_verbs = [
            "analyze", "create", "manage", "plan", "design", "develop", 
            "implement", "organize", "prepare", "build", "extract", 
            "identify", "find", "summarize", "review", "generate",
            "compile", "gather", "collect", "process", "evaluate"
        ]
        
        task_lower = task.lower()
        found_verbs = [verb for verb in common_verbs if verb in task_lower]
        return found_verbs
    
    def calculate_relevance(self, text: str, analysis: Dict[str, Any]) -> float:
        """Calculate relevance score for text - IMPROVED VERSION"""
        if not text or len(text.strip()) < 10:
            return 0.0
            
        text_lower = text.lower()
        score = 0.0
        
        # Persona keywords (30% weight)
        persona_keywords = analysis.get("persona_keywords", [])
        if persona_keywords:
            persona_matches = sum(1 for word in persona_keywords if word in text_lower)
            score += (persona_matches / len(persona_keywords)) * 0.3
        
        # Task keywords (25% weight)
        task_keywords = analysis.get("task_keywords", [])
        if task_keywords:
            task_matches = sum(1 for word in task_keywords if word in text_lower)
            score += (task_matches / len(task_keywords)) * 0.25
        
        # Domain keywords (25% weight)
        domain_keywords = analysis.get("domain_keywords", [])
        if domain_keywords:
            domain_matches = sum(1 for word in domain_keywords if word in text_lower)
            score += (domain_matches / len(domain_keywords)) * 0.25
        
        # Action verbs (10% weight)
        action_verbs = analysis.get("action_verbs", [])
        if action_verbs:
            action_matches = sum(1 for verb in action_verbs if verb in text_lower)
            score += (action_matches / len(action_verbs)) * 0.1
        
        # Length and quality bonus (10% weight)
        length_score = min(len(text) / 200, 1.0) * 0.1
        score += length_score
        
        # If no specific matches, give a base score for meaningful content
        if score == 0 and len(text) > 50:
            score = 0.1  # Base relevance for any substantial text
        
        return min(score, 1.0)
    
    def process_pdf_collection(self, uploaded_files: List, persona: str, task: str) -> Dict[str, Any]:
        """Process a collection of PDF files - IMPROVED VERSION"""
        
        if len(uploaded_files) < 2:
            st.error("❌ Challenge 1B requires at least 2 PDF files. Please upload multiple documents.")
            return {}
        
        # Analyze persona and task
        analysis = self.analyze_persona_and_task(persona, task)
        
        # Debug info
        st.info(f"🧠 Processing {len(uploaded_files)} documents for {analysis['domain']} domain...")
        with st.expander("🔍 Debug: Analysis Configuration"):
            st.write("**Persona Keywords:**", analysis['persona_keywords'])
            st.write("**Task Keywords:**", analysis['task_keywords'])
            st.write("**Domain Keywords:**", analysis['domain_keywords'])
            st.write("**Action Verbs:**", analysis['action_verbs'])
        
        all_sections = []
        all_content = []
        processing_stats = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each PDF
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            sections, content, stats = self._process_single_pdf(uploaded_file, analysis)
            all_sections.extend(sections)
            all_content.extend(content)
            processing_stats[uploaded_file.name] = stats
        
        # Debug processing stats
        with st.expander("📊 Debug: Processing Statistics"):
            for filename, stats in processing_stats.items():
                st.write(f"**{filename}:**")
                st.write(f"  - Pages processed: {stats['pages']}")
                st.write(f"  - Text blocks found: {stats['blocks']}")
                st.write(f"  - Meaningful blocks: {stats['meaningful_blocks']}")
                st.write(f"  - Sections extracted: {stats['sections']}")
                st.write(f"  - Content items: {stats['content_items']}")
        
        # Filter out very low relevance content
        filtered_sections = [s for s in all_sections if s["relevance_score"] > 0.05]
        filtered_content = [c for c in all_content if c["relevance_score"] > 0.05]
        
        # Rank by relevance
        filtered_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        filtered_content.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Assign ranks
        for i, section in enumerate(filtered_sections, 1):
            section["importance_rank"] = i
        
        progress_bar.progress(1.0)
        status_text.text("✅ Collection analysis complete!")
        
        if not filtered_sections and not filtered_content:
            st.warning("⚠️ No relevant content found. Try adjusting your persona or task description to be more specific.")
        
        return {
            "metadata": {
                "input_documents": [f.name for f in uploaded_files],
                "persona": persona,
                "job_to_be_done": task,
                "domain": analysis["domain"],
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_extracted": len(filtered_sections),
                "total_subsections": len(filtered_content),
                "persona_analysis": analysis,
                "processing_stats": processing_stats
            },
            "extracted_sections": filtered_sections[:50],  # Top 50
            "subsection_analysis": filtered_content[:100]  # Top 100
        }
    
    def _process_single_pdf(self, uploaded_file, analysis: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """Process a single PDF file - IMPROVED VERSION"""
        sections = []
        content = []
        stats = {
            "pages": 0,
            "blocks": 0,
            "meaningful_blocks": 0,
            "sections": 0,
            "content_items": 0
        }
        
        try:
            # Read PDF
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            stats["pages"] = len(doc)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try multiple text extraction methods
                text_blocks = []
                
                # Method 1: Dictionary extraction (structured)
                try:
                    text_dict = page.get_text("dict")
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            block_text = self._extract_text_from_block(block)
                            if block_text:
                                text_blocks.append({
                                    "text": block_text,
                                    "method": "dict",
                                    "block": block
                                })
                except:
                    pass
                
                # Method 2: Simple text extraction (fallback)
                if not text_blocks:
                    try:
                        simple_text = page.get_text()
                        if simple_text:
                            # Split into paragraphs
                            paragraphs = [p.strip() for p in simple_text.split('\n\n') if p.strip()]
                            for para in paragraphs:
                                if len(para) > 20:  # Only meaningful paragraphs
                                    text_blocks.append({
                                        "text": para,
                                        "method": "simple",
                                        "block": None
                                    })
                    except:
                        pass
                
                stats["blocks"] += len(text_blocks)
                
                # Process each text block
                for text_block in text_blocks:
                    block_text = text_block["text"]
                    
                    if self._is_meaningful_text(block_text):
                        stats["meaningful_blocks"] += 1
                        
                        relevance_score = self.calculate_relevance(block_text, analysis)
                        
                        if relevance_score > 0.05:  # Lower threshold
                            
                            # Check if it's a section title
                            is_title = False
                            if text_block["method"] == "dict" and text_block["block"]:
                                is_title = self._is_section_title(text_block["block"], block_text)
                            else:
                                # Simple heuristic for titles
                                is_title = (
                                    len(block_text) < 100 and 
                                    len(block_text.split()) < 15 and
                                    not block_text.endswith('.')
                                )
                            
                            if is_title and relevance_score > 0.1:
                                sections.append({
                                    "document": uploaded_file.name,
                                    "section_title": block_text[:150],  # Limit length
                                    "page_number": page_num + 1,
                                    "relevance_score": relevance_score
                                })
                                stats["sections"] += 1
                            
                            # Add to content analysis
                            content.append({
                                "document": uploaded_file.name,
                                "refined_text": block_text,
                                "page_number": page_num + 1,
                                "relevance_score": relevance_score,
                                "relevance_category": self._categorize_relevance(relevance_score)
                            })
                            stats["content_items"] += 1
            
            doc.close()
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        
        return sections, content, stats
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from a block - IMPROVED VERSION"""
        text_parts = []
        try:
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text:
                        line_text += span_text
                if line_text.strip():
                    text_parts.append(line_text.strip())
            
            result = " ".join(text_parts).strip()
            # Clean up extra whitespace
            result = re.sub(r'\s+', ' ', result)
            return result
        except:
            return ""
    
    def _is_meaningful_text(self, text: str) -> bool:
        """Check if text is meaningful - IMPROVED VERSION"""
        if not text or len(text.strip()) < 5:
            return False
        
        # Remove very short text
        if len(text.strip()) < 10:
            return False
        
        # Remove page numbers and headers
        if re.match(r'^\d+$', text.strip()):
            return False
        
        if re.match(r'^Page \d+', text.strip()):
            return False
        
        # Must contain some letters
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Remove pure punctuation
        if re.match(r'^[^\w]*$', text.strip()):
            return False
        
        return True
    
    def _is_section_title(self, block: Dict, text: str) -> bool:
        """Check if text is likely a section title - IMPROVED VERSION"""
        try:
            # Calculate average font size
            total_size = 0
            font_count = 0
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 12)
                    total_size += size
                    font_count += 1
            
            avg_font_size = total_size / font_count if font_count > 0 else 12
            
            # Title heuristics
            is_title = (
                len(text) < 150 and  # Not too long
                len(text.split()) < 20 and  # Not too many words
                avg_font_size > 11 and  # Reasonable font size
                not text.endswith('.') and  # Titles don't end with periods
                len(text.strip()) > 5  # Not too short
            )
            
            return is_title
        except:
            # Fallback heuristic
            return (
                len(text) < 100 and 
                len(text.split()) < 15 and
                not text.endswith('.')
            )
    
    def _categorize_relevance(self, score: float) -> str:
        """Categorize relevance score"""
        if score >= 0.5:
            return "high"
        elif score >= 0.2:
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
        2. **Upload** multiple PDF files (2+ required)
        3. **Analyze** content across all documents
        4. **Get** ranked, relevant results
        """)
        
        st.markdown("## ✨ Features")
        st.markdown("""
        - 🧠 Adaptive to any persona
        - 📚 Multi-document analysis
        - 🎯 Relevance-based ranking
        - 📊 Interactive results
        - 🔍 Debug information
        """)
        
        st.markdown("## 💡 Tips")
        st.markdown("""
        - Be specific with your persona
        - Describe your task clearly
        - Upload related documents
        - Check debug info if no results
        """)
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Your Persona")
        persona = st.text_input(
            "Who are you?",
            placeholder="e.g., Food Contractor, Travel Planner, HR Manager",
            help="Enter your professional role"
        )
    
    with col2:
        st.markdown("#### 🎯 Your Task")
        task = st.text_area(
            "What do you need to do?",
            placeholder="e.g., Plan vegetarian buffet menu for corporate event",
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
            total_keywords = len(analysis['persona_keywords']) + len(analysis['task_keywords']) + len(analysis['domain_keywords'])
            st.write(f"**Keywords:** {total_keywords}")
        with col3:
            st.write(f"**Actions:** {', '.join(analysis['action_verbs'][:3]) if analysis['action_verbs'] else 'General'}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # File upload - MULTIPLE FILES
        st.markdown("#### 📚 Upload Your Document Collection")
        uploaded_files = st.file_uploader(
            "Select multiple PDF files (minimum 2 required)",
            type="pdf",
            accept_multiple_files=True,
            help="Upload 2 or more related PDF documents for analysis"
        )
        
        if uploaded_files:
            if len(uploaded_files) >= 2:
                st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
                
                # Show file details
                with st.expander("📋 Collection Details"):
                    total_size = 0
                    for i, file in enumerate(uploaded_files, 1):
                        size_mb = file.size / (1024 * 1024)
                        total_size += size_mb
                        st.write(f"{i}. **{file.name}** ({size_mb:.1f} MB)")
                    st.write(f"**Total Size:** {total_size:.1f} MB")
                
                # Analyze button
                if st.button("🚀 Analyze Collection", type="primary", use_container_width=True):
                    with st.spinner("🧠 Analyzing your document collection..."):
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
        
        # Show processing stats
        processing_stats = metadata.get('processing_stats', {})
        if processing_stats:
            with st.expander("📈 Processing Summary"):
                for filename, stats in processing_stats.items():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**{filename}**")
                    with col2:
                        st.write(f"Pages: {stats['pages']}, Blocks: {stats['meaningful_blocks']}")
                    with col3:
                        st.write(f"Sections: {stats['sections']}, Content: {stats['content_items']}")
        
        # Tabs for results
        tab1, tab2, tab3 = st.tabs(["🏆 Top Sections", "📝 Content Analysis", "💾 Export"])
        
        with tab1:
            sections = result.get('extracted_sections', [])
            if sections:
                st.markdown(f"### 🏆 Top {len(sections)} Most Relevant Sections")
                
                # Create DataFrame
                df_data = []
                for section in sections:
                    df_data.append({
                        'Rank': section.get('importance_rank', 0),
                        'Document': section.get('document', ''),
                        'Section': section.get('section_title', '')[:80] + "..." if len(section.get('section_title', '')) > 80 else section.get('section_title', ''),
                        'Page': section.get('page_number', 0),
                        'Relevance': f"{section.get('relevance_score', 0):.3f}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Visualization
                if len(df) > 0:
                    doc_counts = df['Document'].value_counts()
                    fig = px.bar(
                        x=doc_counts.index,
                        y=doc_counts.values,
                        title="Sections per Document",
                        labels={'x': 'Document', 'y': 'Number of Sections'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No sections found. Try being more specific with your persona and task.")
        
        with tab2:
            content = result.get('subsection_analysis', [])
            if content:
                st.markdown(f"### 📝 Content Analysis ({len(content)} items)")
                
                # Relevance distribution
                categories = [item.get('relevance_category', 'low') for item in content]
                category_counts = Counter(categories)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔥 High Relevance", category_counts.get('high', 0))
                with col2:
                    st.metric("⚡ Medium Relevance", category_counts.get('medium', 0))
                with col3:
                    st.metric("💡 Low Relevance", category_counts.get('low', 0))
                
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
                        for i, item in enumerate(items[:8], 1):  # Show top 8 per doc
                            category = item.get('relevance_category', 'low')
                            score = item.get('relevance_score', 0)
                            
                            if category == 'high':
                                st.markdown('<div class="relevance-high">', unsafe_allow_html=True)
                                st.markdown("🔥 **HIGH RELEVANCE**")
                            elif category == 'medium':
                                st.markdown('<div class="relevance-medium">', unsafe_allow_html=True)
                                st.markdown("⚡ **MEDIUM RELEVANCE**")
                            else:
                                st.markdown('<div class="relevance-low">', unsafe_allow_html=True)
                                st.markdown("💡 **LOW RELEVANCE**")
                            
                            st.markdown(f"**Item {i} (Page {item['page_number']})**")
                            st.progress(score, text=f"Relevance Score: {score:.3f}")
                            
                            text = item['refined_text']
                            display_text = text[:400] + "..." if len(text) > 400 else text
                            st.write(display_text)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No content found. Try adjusting your persona or task to be more specific.")
        
        with tab3:
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export JSON
                if st.button("📄 Export Complete Analysis (JSON)", use_container_width=True):
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
                    if st.button("📊 Export Sections (CSV)", use_container_width=True):
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
