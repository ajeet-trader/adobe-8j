"""
Adobe India Hackathon 2025 - Challenge 1B: Multi-Collection PDF Analysis
Streamlit Application for Persona-Based Content Analysis
"""

import streamlit as st
import json
import os
from pathlib import Path
import fitz  # PyMuPDF
import re
from datetime import datetime
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
import zipfile
import io

# Page configuration
st.set_page_config(
    page_title="Adobe Challenge 1B - Multi-Collection PDF Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .persona-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .collection-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF0000;
        margin: 0.5rem 0;
    }
    .metric-highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class PersonaBasedAnalyzer:
    def __init__(self):
        """Initialize the persona-based PDF analyzer"""
        
        # Enhanced persona-specific keywords with weights
        self.persona_keywords = {
            "Travel Planner": {
                "high_priority": ["itinerary", "accommodation", "hotel", "restaurant", "attraction", "activity", "booking", "reservation", "transport", "flight"],
                "medium_priority": ["destination", "sightseeing", "tour", "guide", "map", "location", "price", "cost", "budget", "recommendation"],
                "low_priority": ["culture", "history", "tradition", "local", "experience", "tip", "advice", "weather", "season", "group"]
            },
            "HR Professional": {
                "high_priority": ["form", "fillable", "onboarding", "compliance", "employee", "workflow", "automation", "template", "signature", "approval"],
                "medium_priority": ["training", "policy", "process", "document", "management", "creation", "editing", "sharing", "security", "access"],
                "low_priority": ["tutorial", "guide", "instruction", "step", "feature", "tool", "adobe", "acrobat", "pdf", "digital"]
            },
            "Food Contractor": {
                "high_priority": ["vegetarian", "buffet", "menu", "recipe", "ingredient", "gluten-free", "catering", "serving", "preparation", "cooking"],
                "medium_priority": ["dinner", "corporate", "gathering", "dish", "meal", "cuisine", "dietary", "nutrition", "portion", "quantity"],
                "low_priority": ["breakfast", "lunch", "snack", "appetizer", "dessert", "beverage", "seasoning", "technique", "equipment", "kitchen"]
            }
        }
        
        # Task-specific keywords for enhanced relevance
        self.task_keywords = {
            "4-day trip": ["day", "itinerary", "schedule", "plan", "route", "timing", "duration", "stay"],
            "10 college friends": ["group", "friends", "college", "young", "budget", "fun", "social", "party"],
            "South of France": ["france", "french", "southern", "mediterranean", "provence", "nice", "cannes", "marseille"],
            "fillable forms": ["fillable", "form", "field", "input", "interactive", "editable", "data", "entry"],
            "onboarding": ["onboarding", "new", "employee", "hire", "orientation", "welcome", "introduction", "setup"],
            "compliance": ["compliance", "regulation", "policy", "legal", "requirement", "standard", "audit", "documentation"],
            "vegetarian buffet": ["vegetarian", "buffet", "plant-based", "vegan", "meat-free", "self-service", "variety"],
            "corporate gathering": ["corporate", "business", "professional", "office", "company", "meeting", "event"],
            "gluten-free": ["gluten-free", "celiac", "wheat-free", "allergen", "dietary", "restriction", "alternative"]
        }
    
    def analyze_collection(self, pdf_files: List, persona: str, task: str, collection_name: str) -> Dict[str, Any]:
        """Analyze a collection of PDFs for a specific persona and task"""
        
        all_extracted_sections = []
        all_subsection_analysis = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file.name}...")
            progress_bar.progress((i + 1) / len(pdf_files))
            
            sections, subsections = self._process_pdf(pdf_file, persona, task)
            all_extracted_sections.extend(sections)
            all_subsection_analysis.extend(subsections)
        
        # Rank sections by importance
        ranked_sections = self._rank_by_importance(all_extracted_sections, persona, task)
        
        # Ensure diverse representation across documents
        balanced_sections = self._balance_document_representation(ranked_sections)
        
        status_text.text("Analysis complete!")
        progress_bar.progress(1.0)
        
        return {
            "metadata": {
                "input_documents": [f.name for f in pdf_files],
                "persona": persona,
                "job_to_be_done": task,
                "collection_name": collection_name,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_extracted": len(balanced_sections),
                "total_subsections": len(all_subsection_analysis)
            },
            "extracted_sections": balanced_sections[:50],  # Top 50 most relevant
            "subsection_analysis": all_subsection_analysis[:100]  # Top 100 subsections
        }
    
    def _process_pdf(self, pdf_file, persona: str, task: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract and analyze content from a single PDF"""
        sections = []
        subsections = []
        
        try:
            # Read PDF file
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with formatting information
                text_dict = page.get_text("dict")
                
                # Process text blocks
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        block_text = self._extract_block_text(block)
                        
                        if self._is_meaningful_text(block_text):
                            # Calculate relevance score
                            relevance_score = self._calculate_relevance(block_text, persona, task)
                            
                            if relevance_score > 0.2:  # Threshold for inclusion
                                # Determine if this is a section title or content
                                is_title = self._is_section_title(block, block_text)
                                
                                if is_title and relevance_score > 0.4:
                                    sections.append({
                                        "document": pdf_file.name,
                                        "section_title": self._clean_section_title(block_text),
                                        "importance_rank": relevance_score,
                                        "page_number": page_num + 1
                                    })
                                
                                # Add to subsection analysis
                                refined_text = self._refine_text_for_persona(block_text, persona, task)
                                if refined_text:
                                    subsections.append({
                                        "document": pdf_file.name,
                                        "refined_text": refined_text,
                                        "page_number": page_num + 1,
                                        "relevance_score": relevance_score
                                    })
            
            doc.close()
            
        except Exception as e:
            st.error(f"Error processing PDF {pdf_file.name}: {e}")
        
        return sections, subsections
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract clean text from a text block"""
        text_parts = []
        
        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                line_text += span["text"]
            text_parts.append(line_text.strip())
        
        return " ".join(text_parts).strip()
    
    def _is_meaningful_text(self, text: str) -> bool:
        """Check if text is meaningful"""
        if len(text.strip()) < 10:
            return False
        
        # Filter out common non-content patterns
        patterns_to_ignore = [
            r'^\d+$',  # Just page numbers
            r'^Page \d+',  # Page headers
            r'^\d+\s*$',  # Just numbers
            r'^[^\w]*$',  # Just punctuation
        ]
        
        for pattern in patterns_to_ignore:
            if re.match(pattern, text.strip()):
                return False
        
        return True
    
    def _is_section_title(self, block: Dict, text: str) -> bool:
        """Determine if a text block is likely a section title"""
        # Check font size (titles are usually larger)
        avg_font_size = 0
        font_count = 0
        
        for line in block["lines"]:
            for span in line["spans"]:
                avg_font_size += span["size"]
                font_count += 1
        
        if font_count > 0:
            avg_font_size /= font_count
        
        # Heuristics for section titles
        is_title = (
            len(text) < 100 and  # Titles are usually short
            avg_font_size > 12 and  # Titles have larger font
            not text.endswith('.') and  # Titles don't end with periods
            len(text.split()) < 15  # Titles have fewer words
        )
        
        return is_title
    
    def _clean_section_title(self, text: str) -> str:
        """Clean and format section title"""
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
        
        return cleaned
    
    def _calculate_relevance(self, text: str, persona: str, task: str) -> float:
        """Calculate relevance score for text based on persona and task"""
        text_lower = text.lower()
        
        # Get persona keywords
        persona_keywords = self.persona_keywords.get(persona, {})
        
        # Calculate persona-based score
        persona_score = 0
        total_persona_weight = 0
        
        for priority, keywords in persona_keywords.items():
            weight = {"high_priority": 3, "medium_priority": 2, "low_priority": 1}[priority]
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            persona_score += matches * weight
            total_persona_weight += len(keywords) * weight
        
        if total_persona_weight > 0:
            persona_score = persona_score / total_persona_weight
        
        # Calculate task-specific score
        task_score = 0
        for task_phrase, keywords in self.task_keywords.items():
            if task_phrase.lower() in task.lower():
                task_matches = sum(1 for keyword in keywords if keyword in text_lower)
                task_score = task_matches / len(keywords)
                break
        
        # Combine scores with weights
        final_score = (persona_score * 0.7) + (task_score * 0.3)
        
        # Boost score for longer, more substantial content
        length_boost = min(len(text) / 500, 0.2)
        final_score += length_boost
        
        return min(final_score, 1.0)
    
    def _refine_text_for_persona(self, text: str, persona: str, task: str) -> str:
        """Refine extracted text to be more relevant for the specific persona"""
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        if persona == "Travel Planner":
            return self._refine_for_travel_planner(cleaned_text, task)
        elif persona == "HR Professional":
            return self._refine_for_hr_professional(cleaned_text, task)
        elif persona == "Food Contractor":
            return self._refine_for_food_contractor(cleaned_text, task)
        
        return cleaned_text if len(cleaned_text) > 20 else None
    
    def _refine_for_travel_planner(self, text: str, task: str) -> str:
        """Refine text for travel planner persona"""
        if any(keyword in text.lower() for keyword in ["book", "visit", "go to", "recommended", "must-see", "cost", "price"]):
            return text
        
        if re.search(r'\d+.*(?:euro|€|hour|day|minute)', text.lower()):
            return text
        
        return text if len(text) > 30 else None
    
    def _refine_for_hr_professional(self, text: str, task: str) -> str:
        """Refine text for HR professional persona"""
        if re.search(r'\d+\.\s|step \d+|first|then|next|finally', text.lower()):
            return text
        
        if any(keyword in text.lower() for keyword in ["create", "add", "insert", "field", "button", "signature"]):
            return text
        
        return text if len(text) > 25 else None
    
    def _refine_for_food_contractor(self, text: str, task: str) -> str:
        """Refine text for food contractor persona"""
        if re.search(r'\d+\s*(?:cup|tbsp|tsp|oz|lb|gram|kg|ml|liter)', text.lower()):
            return text
        
        if any(keyword in text.lower() for keyword in ["cook", "bake", "fry", "boil", "mix", "add", "serve", "prepare"]):
            return text
        
        return text if len(text) > 20 else None
    
    def _rank_by_importance(self, sections: List[Dict], persona: str, task: str) -> List[Dict]:
        """Rank sections by importance for the given persona and task"""
        sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        for i, section in enumerate(sections, 1):
            section["importance_rank"] = i
        
        return sections
    
    def _balance_document_representation(self, sections: List[Dict]) -> List[Dict]:
        """Ensure balanced representation across all documents"""
        doc_sections = defaultdict(list)
        for section in sections:
            doc_sections[section["document"]].append(section)
        
        balanced = []
        max_per_doc = max(3, len(sections) // len(doc_sections))
        
        for doc, doc_section_list in doc_sections.items():
            balanced.extend(doc_section_list[:max_per_doc])
        
        balanced.sort(key=lambda x: x["importance_rank"])
        
        for i, section in enumerate(balanced, 1):
            section["importance_rank"] = i
        
        return balanced

def initialize_session_state():
    """Initialize session state variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PersonaBasedAnalyzer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None

def display_header():
    """Display the main header"""
    st.markdown('<div class="main-header">📚 Challenge 1B: Multi-Collection PDF Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Persona-Based Content Analysis Across Document Collections</div>', unsafe_allow_html=True)

def display_collection_setup():
    """Display collection setup interface"""
    st.markdown("## 🎯 Collection Setup")
    
    # Predefined collections
    collections = {
        "Travel Planning": {
            "persona": "Travel Planner",
            "task": "Plan a 4-day trip for 10 college friends to South of France",
            "description": "Analyze travel guides for itinerary planning, accommodations, and activities",
            "expected_docs": ["Cities.pdf", "Cuisine.pdf", "History.pdf", "Restaurants & Hotels.pdf", "Things to Do.pdf", "Tips & Tricks.pdf", "Traditions & Culture.pdf"]
        },
        "HR Forms Management": {
            "persona": "HR Professional", 
            "task": "Create and manage fillable forms for onboarding and compliance",
            "description": "Process Adobe Acrobat tutorials for form creation and workflow automation",
            "expected_docs": ["Creating Fillable Forms.pdf", "Form Field Properties.pdf", "Digital Signatures.pdf", "Document Security.pdf", "Workflow Automation.pdf"]
        },
        "Corporate Catering": {
            "persona": "Food Contractor",
            "task": "Prepare vegetarian buffet-style dinner menu for corporate gathering with gluten-free items",
            "description": "Analyze recipe collections for vegetarian and gluten-free catering options",
            "expected_docs": ["Breakfast Recipes.pdf", "Lunch Mains.pdf", "Dinner Mains.pdf", "Side Dishes.pdf", "Gluten-Free Options.pdf", "Vegetarian Specialties.pdf"]
        }
    }
    
    # Collection selection
    selected_collection = st.selectbox(
        "Choose a collection to analyze:",
        list(collections.keys()),
        help="Select one of the predefined collections for analysis"
    )
    
    if selected_collection:
        collection_info = collections[selected_collection]
        
        # Display collection info
        st.markdown(f'<div class="collection-card">', unsafe_allow_html=True)
        st.markdown(f"**Collection:** {selected_collection}")
        st.markdown(f"**Persona:** {collection_info['persona']}")
        st.markdown(f"**Task:** {collection_info['task']}")
        st.markdown(f"**Description:** {collection_info['description']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # File upload
        st.markdown("### 📄 Upload PDF Documents")
        uploaded_files = st.file_uploader(
            f"Upload PDFs for {selected_collection}",
            type="pdf",
            accept_multiple_files=True,
            help=f"Expected documents: {', '.join(collection_info['expected_docs'])}"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} PDF files uploaded successfully!")
            
            # Display uploaded files
            with st.expander("📋 Uploaded Files"):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size:,} bytes)")
            
            # Analyze button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Analyze Collection", type="primary", use_container_width=True):
                    with st.spinner("Analyzing collection... This may take a few minutes."):
                        result = st.session_state.analyzer.analyze_collection(
                            uploaded_files,
                            collection_info['persona'],
                            collection_info['task'],
                            selected_collection
                        )
                        
                        st.session_state.analysis_results[selected_collection] = result
                        st.session_state.current_collection = selected_collection
                        st.success("✅ Analysis completed successfully!")
                        st.rerun()

def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.current_collection or st.session_state.current_collection not in st.session_state.analysis_results:
        return
    
    collection_name = st.session_state.current_collection
    result = st.session_state.analysis_results[collection_name]
    
    st.markdown(f"## 📊 Analysis Results: {collection_name}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🏆 Top Sections", 
        "📝 Content Analysis", 
        "📊 Visualizations",
        "💾 Export"
    ])
    
    with tab1:
        display_overview_tab(result)
    
    with tab2:
        display_sections_tab(result)
    
    with tab3:
        display_content_tab(result)
    
    with tab4:
        display_visualizations_tab(result)
    
    with tab5:
        display_export_tab(result, collection_name)

def display_overview_tab(result):
    """Display overview metrics"""
    metadata = result.get('metadata', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Documents Processed",
            value=len(metadata.get('input_documents', []))
        )
    
    with col2:
        st.metric(
            label="🏆 Sections Extracted",
            value=metadata.get('total_sections_extracted', 0)
        )
    
    with col3:
        st.metric(
            label="📝 Subsections Analyzed",
            value=metadata.get('total_subsections', 0)
        )
    
    with col4:
        st.metric(
            label="👤 Persona",
            value=metadata.get('persona', 'Unknown')
        )
    
    # Collection details
    st.markdown("### 🎯 Collection Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Processing Information**")
        st.write(f"**Collection:** {metadata.get('collection_name', 'Unknown')}")
        st.write(f"**Persona:** {metadata.get('persona', 'Unknown')}")
        st.write(f"**Processing Time:** {metadata.get('processing_timestamp', 'Unknown')}")
    
    with col2:
        st.markdown("**📚 Input Documents**")
        for doc in metadata.get('input_documents', []):
            st.write(f"• {doc}")
    
    # Task description
    st.markdown("### 🎯 Task Description")
    st.info(metadata.get('job_to_be_done', 'No task description available'))

def display_sections_tab(result):
    """Display top extracted sections"""
    sections = result.get('extracted_sections', [])
    
    if not sections:
        st.warning("No sections were extracted from the documents.")
        return
    
    st.markdown(f"### 🏆 Top {len(sections)} Most Relevant Sections")
    
    # Create DataFrame for better display
    sections_data = []
    for section in sections:
        sections_data.append({
            'Rank': section.get('importance_rank', 0),
            'Document': section.get('document', ''),
            'Section Title': section.get('section_title', ''),
            'Page': section.get('page_number', 0)
        })
    
    df_sections = pd.DataFrame(sections_data)
    
    # Display with filtering options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Filter by document
        unique_docs = df_sections['Document'].unique()
        selected_docs = st.multiselect(
            "Filter by Document:",
            unique_docs,
            default=unique_docs
        )
        
        # Filter by rank range
        max_rank = df_sections['Rank'].max()
        rank_range = st.slider(
            "Rank Range:",
            1, max_rank,
            (1, min(20, max_rank))
        )
    
    with col2:
        # Apply filters
        filtered_df = df_sections[
            (df_sections['Document'].isin(selected_docs)) &
            (df_sections['Rank'] >= rank_range[0]) &
            (df_sections['Rank'] <= rank_range[1])
        ]
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Section details
    if not filtered_df.empty:
        st.markdown("### 🔍 Section Details")
        
        selected_rank = st.selectbox(
            "Select a section to view details:",
            filtered_df['Rank'].tolist(),
            format_func=lambda x: f"Rank {x}: {filtered_df[filtered_df['Rank']==x]['Section Title'].iloc[0][:50]}..."
        )
        
        if selected_rank:
            selected_section = next(s for s in sections if s['importance_rank'] == selected_rank)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Document:** {selected_section['document']}")
            with col2:
                st.write(f"**Page:** {selected_section['page_number']}")
            with col3:
                st.write(f"**Rank:** {selected_section['importance_rank']}")
            
            st.markdown("**Section Title:**")
            st.write(selected_section['section_title'])

def display_content_tab(result):
    """Display content analysis"""
    subsections = result.get('subsection_analysis', [])
    
    if not subsections:
        st.warning("No content analysis available.")
        return
    
    st.markdown(f"### 📝 Content Analysis ({len(subsections)} items)")
    
    # Group by document
    doc_content = defaultdict(list)
    for subsection in subsections:
        doc_content[subsection['document']].append(subsection)
    
    # Display content by document
    for doc_name, content_list in doc_content.items():
        with st.expander(f"📄 {doc_name} ({len(content_list)} items)"):
            
            # Sort by relevance score if available
            if 'relevance_score' in content_list[0]:
                content_list.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            for i, content in enumerate(content_list[:10]):  # Show top 10 per document
                st.markdown(f"**Item {i+1} (Page {content['page_number']})**")
                
                # Show relevance score if available
                if 'relevance_score' in content:
                    score = content['relevance_score']
                    st.progress(score, text=f"Relevance Score: {score:.3f}")
                
                st.write(content['refined_text'][:500] + "..." if len(content['refined_text']) > 500 else content['refined_text'])
                st.markdown("---")

def display_visualizations_tab(result):
    """Display visualizations"""
    sections = result.get('extracted_sections', [])
    subsections = result.get('subsection_analysis', [])
    
    if not sections and not subsections:
        st.warning("No data available for visualization.")
        return
    
    # Document distribution chart
    if sections:
        st.markdown("### 📊 Section Distribution by Document")
        
        doc_counts = defaultdict(int)
        for section in sections:
            doc_counts[section['document']] += 1
        
        fig_pie = px.pie(
            values=list(doc_counts.values()),
            names=list(doc_counts.keys()),
            title="Sections per Document"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Importance ranking visualization
    if sections:
        st.markdown("### 📈 Importance Ranking Distribution")
        
        # Create ranking bins
        ranks = [s['importance_rank'] for s in sections]
        
        fig_hist = px.histogram(
            x=ranks,
            nbins=20,
            title="Distribution of Section Rankings",
            labels={'x': 'Importance Rank', 'y': 'Number of Sections'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Page distribution
    if subsections:
        st.markdown("### 📄 Content Distribution by Page")
        
        page_counts = defaultdict(int)
        for subsection in subsections:
            page_counts[subsection['page_number']] += 1
        
        pages = sorted(page_counts.keys())
        counts = [page_counts[p] for p in pages]
        
        fig_line = px.line(
            x=pages,
            y=counts,
            title="Content Items per Page",
            labels={'x': 'Page Number', 'y': 'Number of Content Items'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Relevance score distribution
    if subsections and 'relevance_score' in subsections[0]:
        st.markdown("### 🎯 Relevance Score Distribution")
        
        scores = [s.get('relevance_score', 0) for s in subsections]
        
        fig_box = px.box(
            y=scores,
            title="Relevance Score Distribution",
            labels={'y': 'Relevance Score'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

def display_export_tab(result, collection_name):
    """Display export options"""
    st.markdown("### 💾 Export Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export complete results as JSON
        if st.button("📄 Export Complete Analysis (JSON)", use_container_width=True):
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"{collection_name.lower().replace(' ', '_')}_analysis.json",
                mime="application/json"
            )
    
    with col2:
        # Export sections as CSV
        sections = result.get('extracted_sections', [])
        if sections and st.button("📊 Export Sections (CSV)", use_container_width=True):
            df = pd.DataFrame(sections)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{collection_name.lower().replace(' ', '_')}_sections.csv",
                mime="text/csv"
            )
    
    with col3:
        # Export summary report
        if st.button("📋 Export Summary Report", use_container_width=True):
            report = generate_summary_report(result, collection_name)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"{collection_name.lower().replace(' ', '_')}_report.md",
                mime="text/markdown"
            )
    
    # Preview export data
    st.markdown("### 👀 Export Preview")
    
    export_type = st.selectbox(
        "Select data to preview:",
        ["Complete Analysis (JSON)", "Sections (CSV)", "Summary Report (Markdown)"]
    )
    
    if export_type == "Complete Analysis (JSON)":
        st.json(result)
    elif export_type == "Sections (CSV)":
        sections = result.get('extracted_sections', [])
        if sections:
            df = pd.DataFrame(sections)
            st.dataframe(df)
    elif export_type == "Summary Report (Markdown)":
        report = generate_summary_report(result, collection_name)
        st.markdown(report)

def generate_summary_report(result, collection_name):
    """Generate a summary report in Markdown format"""
    metadata = result.get('metadata', {})
    sections = result.get('extracted_sections', [])
    
    report = f"""# {collection_name} - Analysis Report

## Collection Overview
- **Persona:** {metadata.get('persona', 'Unknown')}
- **Task:** {metadata.get('job_to_be_done', 'Unknown')}
- **Documents Processed:** {len(metadata.get('input_documents', []))}
- **Processing Date:** {metadata.get('processing_timestamp', 'Unknown')}

## Analysis Results
- **Total Sections Extracted:** {metadata.get('total_sections_extracted', 0)}
- **Total Subsections Analyzed:** {metadata.get('total_subsections', 0)}

## Input Documents
"""
    
    for doc in metadata.get('input_documents', []):
        report += f"- {doc}\n"
    
    report += "\n## Top 10 Most Relevant Sections\n\n"
    
    for i, section in enumerate(sections[:10], 1):
        report += f"{i}. **{section.get('section_title', 'Unknown')}**\n"
        report += f"   - Document: {section.get('document', 'Unknown')}\n"
        report += f"   - Page: {section.get('page_number', 'Unknown')}\n"
        report += f"   - Importance Rank: {section.get('importance_rank', 'Unknown')}\n\n"
    
    report += f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    
    return report

def main():
    """Main application function"""
    initialize_session_state()
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Challenge 1B")
        st.markdown("**Multi-Collection PDF Analysis**")
        
        st.markdown("### 🚀 Features")
        st.markdown("""
        - 👤 Persona-based analysis
        - 🏆 Importance ranking
        - 📊 Interactive visualizations
        - 💾 Multiple export formats
        - 📈 Content quality metrics
        """)
        
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Select a collection type
        2. Upload PDF documents
        3. Click 'Analyze Collection'
        4. Explore results in tabs
        5. Export your findings
        """)
        
        # Reset button
        if st.button("🔄 Reset Analysis", type="secondary"):
            st.session_state.analysis_results = {}
            st.session_state.current_collection = None
            st.rerun()
    
    # Main content
    if not st.session_state.current_collection:
        display_collection_setup()
    else:
        display_analysis_results()
        
        # Option to analyze another collection
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📚 Analyze Another Collection", type="secondary", use_container_width=True):
                st.session_state.current_collection = None
                st.rerun()

if __name__ == "__main__":
    main()
