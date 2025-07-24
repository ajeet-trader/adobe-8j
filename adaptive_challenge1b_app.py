"""
Adobe India Hackathon 2025 - Challenge 1B: Adaptive Smart PDF Tool
Dynamic persona-based content analysis that adapts to ANY persona and task
"""

import streamlit as st
import json
import os
from pathlib import Path
import fitz  # PyMuPDF
import re
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from collection_manager import integrate_collection_manager

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Adobe Challenge 1B - Adaptive Smart PDF Tool",
    page_icon="🧠",
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
        background: linear-gradient(90deg, #FF0000, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .adaptive-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .dynamic-persona {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .task-analysis {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .relevance-high { background-color: #d4edda; padding: 0.5rem; border-radius: 5px; }
    .relevance-medium { background-color: #fff3cd; padding: 0.5rem; border-radius: 5px; }
    .relevance-low { background-color: #f8d7da; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

class AdaptivePersonaAnalyzer:
    def __init__(self):
        """Initialize the adaptive analyzer with dynamic capabilities"""
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Load spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy model not found. Some advanced features may be limited.")
            self.nlp = None
    
    def analyze_persona_and_task(self, persona: str, task: str) -> Dict[str, Any]:
        """
        Dynamically analyze any persona and task to extract key characteristics
        """
        analysis = {
            "persona_keywords": [],
            "task_keywords": [],
            "domain_indicators": [],
            "action_verbs": [],
            "priority_concepts": [],
            "context_clues": []
        }
        
        # Analyze persona
        persona_tokens = self._extract_meaningful_tokens(persona.lower())
        analysis["persona_keywords"] = persona_tokens
        
        # Analyze task
        task_tokens = self._extract_meaningful_tokens(task.lower())
        analysis["task_keywords"] = task_tokens
        
        # Extract action verbs from task
        if self.nlp:
            doc = self.nlp(task)
            analysis["action_verbs"] = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        else:
            # Simple verb extraction
            common_verbs = ["create", "manage", "plan", "analyze", "design", "develop", "implement", "organize", "prepare", "build"]
            analysis["action_verbs"] = [verb for verb in common_verbs if verb in task.lower()]
        
        # Identify domain indicators
        analysis["domain_indicators"] = self._identify_domain(persona, task)
        
        # Extract priority concepts (nouns and important terms)
        if self.nlp:
            doc = self.nlp(f"{persona} {task}")
            analysis["priority_concepts"] = [
                token.lemma_ for token in doc 
                if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2
            ]
        
        # Context clues (numbers, time references, etc.)
        analysis["context_clues"] = self._extract_context_clues(task)
        
        return analysis
    
    def _extract_meaningful_tokens(self, text: str) -> List[str]:
        """Extract meaningful tokens from text"""
        tokens = word_tokenize(text)
        meaningful_tokens = [
            token.lower() for token in tokens 
            if token.isalpha() and len(token) > 2 and token.lower() not in self.stop_words
        ]
        return list(set(meaningful_tokens))
    
    def _identify_domain(self, persona: str, task: str) -> List[str]:
        """Identify the domain/industry based on persona and task"""
        domain_keywords = {
            "technology": ["software", "app", "system", "digital", "tech", "IT", "developer", "engineer"],
            "business": ["management", "strategy", "corporate", "business", "executive", "manager"],
            "healthcare": ["medical", "health", "patient", "clinical", "doctor", "nurse", "healthcare"],
            "education": ["student", "teacher", "academic", "education", "learning", "school", "university"],
            "finance": ["financial", "accounting", "budget", "investment", "banking", "finance"],
            "marketing": ["marketing", "brand", "campaign", "advertising", "promotion", "social media"],
            "travel": ["travel", "tourism", "trip", "vacation", "destination", "hotel", "flight"],
            "food": ["food", "recipe", "cooking", "restaurant", "chef", "culinary", "menu"],
            "legal": ["legal", "law", "attorney", "compliance", "regulation", "contract"],
            "hr": ["human resources", "HR", "employee", "recruitment", "onboarding", "training"]
        }
        
        text = f"{persona} {task}".lower()
        identified_domains = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                identified_domains.append(domain)
        
        return identified_domains
    
    def _extract_context_clues(self, task: str) -> List[str]:
        """Extract contextual information like numbers, dates, etc."""
        context_clues = []
        
        # Numbers
        numbers = re.findall(r'\d+', task)
        context_clues.extend([f"number_{num}" for num in numbers])
        
        # Time references
        time_words = ["day", "week", "month", "year", "daily", "weekly", "monthly", "annual"]
        for word in time_words:
            if word in task.lower():
                context_clues.append(f"time_{word}")
        
        # Size/scale indicators
        scale_words = ["small", "large", "big", "massive", "tiny", "huge", "enterprise", "startup"]
        for word in scale_words:
            if word in task.lower():
                context_clues.append(f"scale_{word}")
        
        return context_clues
    
    def calculate_adaptive_relevance(self, text: str, persona_analysis: Dict[str, Any]) -> float:
        """
        Calculate relevance score dynamically based on persona and task analysis
        """
        text_lower = text.lower()
        total_score = 0.0
        max_possible_score = 0.0
        
        # Persona keyword matching (30% weight)
        persona_keywords = persona_analysis.get("persona_keywords", [])
        if persona_keywords:
            persona_matches = sum(1 for keyword in persona_keywords if keyword in text_lower)
            persona_score = persona_matches / len(persona_keywords)
            total_score += persona_score * 0.3
        max_possible_score += 0.3
        
        # Task keyword matching (25% weight)
        task_keywords = persona_analysis.get("task_keywords", [])
        if task_keywords:
            task_matches = sum(1 for keyword in task_keywords if keyword in text_lower)
            task_score = task_matches / len(task_keywords)
            total_score += task_score * 0.25
        max_possible_score += 0.25
        
        # Action verb matching (20% weight)
        action_verbs = persona_analysis.get("action_verbs", [])
        if action_verbs:
            action_matches = sum(1 for verb in action_verbs if verb in text_lower)
            action_score = action_matches / len(action_verbs) if action_verbs else 0
            total_score += action_score * 0.2
        max_possible_score += 0.2
        
        # Priority concepts matching (15% weight)
        priority_concepts = persona_analysis.get("priority_concepts", [])
        if priority_concepts:
            concept_matches = sum(1 for concept in priority_concepts if concept in text_lower)
            concept_score = concept_matches / len(priority_concepts)
            total_score += concept_score * 0.15
        max_possible_score += 0.15
        
        # Context clues matching (10% weight)
        context_clues = persona_analysis.get("context_clues", [])
        if context_clues:
            context_matches = sum(1 for clue in context_clues if clue.split('_')[1] in text_lower)
            context_score = context_matches / len(context_clues)
            total_score += context_score * 0.1
        max_possible_score += 0.1
        
        # Normalize score
        if max_possible_score > 0:
            normalized_score = total_score / max_possible_score
        else:
            normalized_score = 0.0
        
        # Content quality boost (length and structure)
        quality_boost = min(len(text) / 1000, 0.2)  # Up to 20% boost
        
        # Domain relevance boost
        domain_boost = 0.0
        domains = persona_analysis.get("domain_indicators", [])
        for domain in domains:
            if any(domain_word in text_lower for domain_word in domain.split()):
                domain_boost += 0.1
        
        final_score = min(normalized_score + quality_boost + domain_boost, 1.0)
        return final_score
    
    def analyze_collection(self, pdf_files: List, persona: str, task: str, collection_name: str) -> Dict[str, Any]:
        """Analyze a collection of PDFs with adaptive persona-based analysis"""
        
        if len(pdf_files) < 2:
            st.error("❌ Challenge 1B requires multiple PDF files as a collection. Please upload at least 2 documents.")
            return {}
        
        # First, analyze the persona and task dynamically
        st.info(f"🧠 Analyzing persona and task characteristics for {len(pdf_files)}-document collection...")
        persona_analysis = self.analyze_persona_and_task(persona, task)
        
        # Display the analysis
        self._display_persona_analysis(persona_analysis, persona, task)
        
        all_extracted_sections = []
        all_subsection_analysis = []
        document_summaries = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each document in the collection
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing document {i+1}/{len(pdf_files)}: {pdf_file.name}")
            progress_bar.progress((i + 1) / len(pdf_files))
            
            sections, subsections = self._process_pdf_adaptive(pdf_file, persona_analysis)
            all_extracted_sections.extend(sections)
            all_subsection_analysis.extend(subsections)
            
            # Create document summary
            document_summaries[pdf_file.name] = {
                "sections_found": len(sections),
                "content_items": len(subsections),
                "avg_relevance": np.mean([s.get('relevance_score', 0) for s in subsections]) if subsections else 0
            }
        
        status_text.text("🔄 Performing cross-document analysis and ranking...")
        
        # Rank sections by adaptive relevance across the entire collection
        ranked_sections = self._rank_by_adaptive_importance(all_extracted_sections, persona_analysis)
        
        # Ensure diverse representation across all documents in collection
        balanced_sections = self._balance_document_representation(ranked_sections)
        
        # Perform cross-document analysis
        cross_document_insights = self._analyze_cross_document_patterns(balanced_sections, all_subsection_analysis)
        
        # Generate insights about the collection analysis
        analysis_insights = self._generate_collection_insights(balanced_sections, persona_analysis, document_summaries)
        
        status_text.text("✅ Collection analysis complete!")
        progress_bar.progress(1.0)
        
        return {
            "metadata": {
                "input_documents": [f.name for f in pdf_files],
                "collection_size": len(pdf_files),
                "persona": persona,
                "job_to_be_done": task,
                "collection_name": collection_name,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_extracted": len(balanced_sections),
                "total_subsections": len(all_subsection_analysis),
                "persona_analysis": persona_analysis,
                "analysis_insights": analysis_insights,
                "document_summaries": document_summaries,
                "cross_document_insights": cross_document_insights
            },
            "extracted_sections": balanced_sections[:50],  # Top 50 across entire collection
            "subsection_analysis": all_subsection_analysis[:100]  # Top 100 across entire collection
        }
    
    def _display_persona_analysis(self, persona_analysis: Dict[str, Any], persona: str, task: str):
        """Display the dynamic persona analysis"""
        with st.expander("🔍 Dynamic Persona & Task Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 👤 Persona Analysis")
                st.write(f"**Role:** {persona}")
                
                if persona_analysis["domain_indicators"]:
                    st.write(f"**Identified Domains:** {', '.join(persona_analysis['domain_indicators'])}")
                
                if persona_analysis["persona_keywords"]:
                    st.write(f"**Key Terms:** {', '.join(persona_analysis['persona_keywords'][:10])}")
            
            with col2:
                st.markdown("### 🎯 Task Analysis")
                st.write(f"**Task:** {task}")
                
                if persona_analysis["action_verbs"]:
                    st.write(f"**Action Verbs:** {', '.join(persona_analysis['action_verbs'])}")
                
                if persona_analysis["context_clues"]:
                    st.write(f"**Context Clues:** {', '.join(persona_analysis['context_clues'])}")
    
    def _process_pdf_adaptive(self, pdf_file, persona_analysis: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """Process PDF with adaptive analysis"""
        sections = []
        subsections = []
        
        try:
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        block_text = self._extract_block_text(block)
                        
                        if self._is_meaningful_text(block_text):
                            # Calculate adaptive relevance score
                            relevance_score = self.calculate_adaptive_relevance(block_text, persona_analysis)
                            
                            if relevance_score > 0.15:  # Lower threshold for adaptive analysis
                                is_title = self._is_section_title(block, block_text)
                                
                                if is_title and relevance_score > 0.3:
                                    sections.append({
                                        "document": pdf_file.name,
                                        "section_title": self._clean_section_title(block_text),
                                        "importance_rank": relevance_score,
                                        "page_number": page_num + 1,
                                        "relevance_factors": self._get_relevance_factors(block_text, persona_analysis)
                                    })
                                
                                refined_text = self._refine_text_adaptive(block_text, persona_analysis)
                                if refined_text:
                                    subsections.append({
                                        "document": pdf_file.name,
                                        "refined_text": refined_text,
                                        "page_number": page_num + 1,
                                        "relevance_score": relevance_score,
                                        "relevance_category": self._categorize_relevance(relevance_score)
                                    })
            
            doc.close()
            
        except Exception as e:
            st.error(f"Error processing PDF {pdf_file.name}: {e}")
        
        return sections, subsections
    
    def _get_relevance_factors(self, text: str, persona_analysis: Dict[str, Any]) -> List[str]:
        """Identify which factors contributed to the relevance score"""
        factors = []
        text_lower = text.lower()
        
        # Check persona keywords
        persona_keywords = persona_analysis.get("persona_keywords", [])
        if any(keyword in text_lower for keyword in persona_keywords):
            factors.append("persona_keywords")
        
        # Check task keywords
        task_keywords = persona_analysis.get("task_keywords", [])
        if any(keyword in text_lower for keyword in task_keywords):
            factors.append("task_keywords")
        
        # Check action verbs
        action_verbs = persona_analysis.get("action_verbs", [])
        if any(verb in text_lower for verb in action_verbs):
            factors.append("action_verbs")
        
        # Check domain indicators
        domains = persona_analysis.get("domain_indicators", [])
        if any(domain in text_lower for domain in domains):
            factors.append("domain_match")
        
        return factors
    
    def _categorize_relevance(self, score: float) -> str:
        """Categorize relevance score"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _refine_text_adaptive(self, text: str, persona_analysis: Dict[str, Any]) -> str:
        """Adaptively refine text based on persona analysis"""
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Apply domain-specific refinement
        domains = persona_analysis.get("domain_indicators", [])
        
        if "technology" in domains:
            return self._refine_for_technology(cleaned_text)
        elif "business" in domains:
            return self._refine_for_business(cleaned_text)
        elif "healthcare" in domains:
            return self._refine_for_healthcare(cleaned_text)
        elif "education" in domains:
            return self._refine_for_education(cleaned_text)
        elif "travel" in domains:
            return self._refine_for_travel(cleaned_text)
        elif "food" in domains:
            return self._refine_for_food(cleaned_text)
        else:
            # Generic refinement
            return self._refine_generic(cleaned_text, persona_analysis)
    
    def _refine_for_technology(self, text: str) -> str:
        """Refine text for technology domain"""
        tech_indicators = ["API", "software", "system", "code", "development", "implementation", "architecture"]
        if any(indicator.lower() in text.lower() for indicator in tech_indicators):
            return text
        return text if len(text) > 25 else None
    
    def _refine_for_business(self, text: str) -> str:
        """Refine text for business domain"""
        business_indicators = ["strategy", "management", "process", "workflow", "efficiency", "ROI", "KPI"]
        if any(indicator.lower() in text.lower() for indicator in business_indicators):
            return text
        return text if len(text) > 30 else None
    
    def _refine_for_healthcare(self, text: str) -> str:
        """Refine text for healthcare domain"""
        health_indicators = ["patient", "treatment", "diagnosis", "medical", "clinical", "therapy"]
        if any(indicator.lower() in text.lower() for indicator in health_indicators):
            return text
        return text if len(text) > 25 else None
    
    def _refine_for_education(self, text: str) -> str:
        """Refine text for education domain"""
        edu_indicators = ["student", "learning", "curriculum", "assessment", "teaching", "academic"]
        if any(indicator.lower() in text.lower() for indicator in edu_indicators):
            return text
        return text if len(text) > 25 else None
    
    def _refine_for_travel(self, text: str) -> str:
        """Refine text for travel domain"""
        travel_indicators = ["destination", "booking", "accommodation", "itinerary", "transport", "attraction"]
        if any(indicator.lower() in text.lower() for indicator in travel_indicators):
            return text
        return text if len(text) > 30 else None
    
    def _refine_for_food(self, text: str) -> str:
        """Refine text for food domain"""
        food_indicators = ["recipe", "ingredient", "cooking", "preparation", "serving", "dietary"]
        if any(indicator.lower() in text.lower() for indicator in food_indicators):
            return text
        return text if len(text) > 20 else None
    
    def _refine_generic(self, text: str, persona_analysis: Dict[str, Any]) -> str:
        """Generic text refinement based on persona analysis"""
        # Look for action verbs from the task
        action_verbs = persona_analysis.get("action_verbs", [])
        if any(verb in text.lower() for verb in action_verbs):
            return text
        
        # Look for priority concepts
        priority_concepts = persona_analysis.get("priority_concepts", [])
        if any(concept in text.lower() for concept in priority_concepts):
            return text
        
        return text if len(text) > 25 else None
    
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
        if len(text.strip()) < 8:
            return False
        
        patterns_to_ignore = [
            r'^\d+$', r'^Page \d+', r'^\d+\s*$', r'^[^\w]*$'
        ]
        
        for pattern in patterns_to_ignore:
            if re.match(pattern, text.strip()):
                return False
        
        return True
    
    def _is_section_title(self, block: Dict, text: str) -> bool:
        """Determine if a text block is likely a section title"""
        avg_font_size = 0
        font_count = 0
        
        for line in block["lines"]:
            for span in line["spans"]:
                avg_font_size += span["size"]
                font_count += 1
        
        if font_count > 0:
            avg_font_size /= font_count
        
        is_title = (
            len(text) < 120 and
            avg_font_size > 11 and
            not text.endswith('.') and
            len(text.split()) < 20
        )
        
        return is_title
    
    def _clean_section_title(self, text: str) -> str:
        """Clean and format section title"""
        cleaned = re.sub(r'\s+', ' ', text.strip())
        if len(cleaned) > 120:
            cleaned = cleaned[:117] + "..."
        return cleaned
    
    def _rank_by_adaptive_importance(self, sections: List[Dict], persona_analysis: Dict[str, Any]) -> List[Dict]:
        """Rank sections by adaptive importance"""
        sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        for i, section in enumerate(sections, 1):
            section["importance_rank"] = i
        
        return sections
    
    def _balance_document_representation(self, sections: List[Dict]) -> List[Dict]:
        """Ensure balanced representation across documents"""
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
    
    def _generate_analysis_insights(self, sections: List[Dict], persona_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about the analysis"""
        insights = {
            "top_relevance_factors": [],
            "document_coverage": {},
            "content_distribution": {},
            "adaptation_summary": ""
        }
        
        # Analyze top relevance factors
        all_factors = []
        for section in sections:
            all_factors.extend(section.get("relevance_factors", []))
        
        factor_counts = Counter(all_factors)
        insights["top_relevance_factors"] = factor_counts.most_common(5)
        
        # Document coverage
        doc_counts = Counter(section["document"] for section in sections)
        insights["document_coverage"] = dict(doc_counts)
        
        # Content distribution by relevance
        relevance_scores = [section["importance_rank"] for section in sections]
        insights["content_distribution"] = {
            "high_relevance": len([s for s in relevance_scores if s <= 10]),
            "medium_relevance": len([s for s in relevance_scores if 10 < s <= 30]),
            "low_relevance": len([s for s in relevance_scores if s > 30])
        }
        
        # Adaptation summary
        domains = persona_analysis.get("domain_indicators", [])
        insights["adaptation_summary"] = f"Adapted analysis for {', '.join(domains) if domains else 'general'} domain(s)"
        
        return insights

    def _analyze_cross_document_patterns(self, sections: List[Dict], subsections: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns across multiple documents in the collection"""
        cross_insights = {
            "document_overlap": {},
            "complementary_content": [],
            "collection_coverage": {},
            "content_gaps": []
        }
        
        # Analyze document overlap in topics
        doc_topics = defaultdict(set)
        for section in sections:
            doc_name = section['document']
            section_words = set(section['section_title'].lower().split())
            doc_topics[doc_name].update(section_words)
        
        # Find overlapping topics between documents
        docs = list(doc_topics.keys())
        for i, doc1 in enumerate(docs):
            for doc2 in docs[i+1:]:
                overlap = doc_topics[doc1] & doc_topics[doc2]
                if overlap:
                    cross_insights["document_overlap"][f"{doc1} ↔ {doc2}"] = list(overlap)
        
        # Identify complementary content
        doc_content_types = defaultdict(list)
        for subsection in subsections:
            doc_name = subsection['document']
            content = subsection['refined_text'].lower()
            
            # Categorize content type
            if any(word in content for word in ['step', 'instruction', 'how to', 'process']):
                doc_content_types[doc_name].append('instructional')
            elif any(word in content for word in ['list', 'items', 'options', 'choices']):
                doc_content_types[doc_name].append('reference')
            elif any(word in content for word in ['example', 'case', 'scenario']):
                doc_content_types[doc_name].append('examples')
        
        cross_insights["complementary_content"] = dict(doc_content_types)
        
        return cross_insights

    def _generate_collection_insights(self, sections: List[Dict], persona_analysis: Dict[str, Any], document_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights specific to collection analysis"""
        insights = {
            "collection_strength": 0.0,
            "document_contribution": {},
            "coverage_analysis": {},
            "collection_recommendations": []
        }
        
        # Calculate collection strength
        if sections:
            avg_relevance = np.mean([s.get('importance_rank', 0) for s in sections])
            insights["collection_strength"] = min(avg_relevance / len(sections), 1.0)
        
        # Document contribution analysis
        doc_contributions = defaultdict(int)
        for section in sections:
            doc_contributions[section['document']] += 1
        
        total_sections = len(sections)
        for doc, count in doc_contributions.items():
            insights["document_contribution"][doc] = {
                "sections": count,
                "percentage": (count / total_sections) * 100 if total_sections > 0 else 0
            }
        
        # Coverage analysis
        domains = persona_analysis.get('domain_indicators', [])
        if domains:
            insights["coverage_analysis"]["domains_covered"] = len(domains)
            insights["coverage_analysis"]["primary_domain"] = domains[0] if domains else "general"
        
        # Generate recommendations
        if len(document_summaries) < 3:
            insights["collection_recommendations"].append("Consider adding more documents to the collection for comprehensive analysis")
        
        # Check for balanced contribution
        contributions = list(doc_contributions.values())
        if contributions and max(contributions) > min(contributions) * 3:
            insights["collection_recommendations"].append("Some documents contribute significantly more content - ensure all documents are relevant to the persona/task")
        
        return insights

def initialize_session_state():
    """Initialize session state variables"""
    if 'adaptive_analyzer' not in st.session_state:
        st.session_state.adaptive_analyzer = AdaptivePersonaAnalyzer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

def display_header():
    """Display the main header"""
    st.markdown('<div class="main-header">🧠 Adaptive Smart PDF Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dynamic Persona-Based Analysis for ANY Use Case</div>', unsafe_allow_html=True)

def display_adaptive_setup():
    """Display the adaptive setup interface"""
    st.markdown('<div class="adaptive-card">', unsafe_allow_html=True)
    st.markdown("## 🎯 Define Your Analysis")
    st.markdown("**This tool adapts to ANY persona and task - no predefined limitations!**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Custom persona and task input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Define Your Persona")
        persona = st.text_input(
            "Enter the persona/role:",
            placeholder="e.g., Data Scientist, Marketing Manager, Legal Advisor, etc.",
            help="Enter any professional role or persona"
        )
        
        # Persona examples
        with st.expander("💡 Persona Examples"):
            st.markdown("""
            - **Data Scientist**: Analyzing research papers for ML insights
            - **Marketing Manager**: Extracting campaign strategies from reports
            - **Legal Advisor**: Finding compliance requirements in regulations
            - **Project Manager**: Identifying project risks and timelines
            - **Financial Analyst**: Extracting investment insights from reports
            - **UX Designer**: Finding user research insights from studies
            - **Sales Director**: Extracting market trends from industry reports
            """)
    
    with col2:
        st.markdown("### 🎯 Define Your Task")
        task = st.text_area(
            "Describe your specific task:",
            placeholder="e.g., Extract key findings for quarterly presentation to executives",
            height=100,
            help="Be specific about what you want to accomplish"
        )
        
        # Task examples
        with st.expander("💡 Task Examples"):
            st.markdown("""
            - **Research Analysis**: "Summarize key findings for literature review"
            - **Compliance Check**: "Identify regulatory requirements for new product launch"
            - **Market Research**: "Extract competitor strategies and pricing models"
            - **Risk Assessment**: "Identify potential risks and mitigation strategies"
            - **Technical Review**: "Extract implementation guidelines and best practices"
            - **Strategic Planning**: "Identify market opportunities and threats"
            """)
    
    # Collection name
    collection_name = st.text_input(
        "Collection Name:",
        placeholder="e.g., Q4 Market Research, Compliance Documents, etc.",
        help="Give your document collection a descriptive name"
    )
    
    # File upload - MULTIPLE FILES AS COLLECTION using Collection Manager
    if persona and task and collection_name:
        st.markdown("### 📚 Create Your Document Collection")
        
        # Initialize collection manager
        collection_manager = integrate_collection_manager()
        
        # Use collection manager interface
        collection_result = collection_manager.create_collection_interface()
        
        if collection_result:
            if collection_result["collection_type"] == "uploaded":
                uploaded_files = collection_result["files"]
                collection_stats = collection_result["collection_stats"]
                
                # Show collection processing strategy
                st.markdown("### 🧠 Collection Processing Strategy")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📊 Cross-Document Analysis**")
                    st.write("• Analyze content across all documents")
                    st.write("• Find connections between different PDFs")
                    st.write("• Ensure balanced representation")
                
                with col2:
                    st.markdown("**🎯 Unified Relevance Ranking**")
                    st.write("• Rank content from entire collection")
                    st.write("• Prioritize based on persona + task")
                    st.write("• Provide collection-wide insights")
                
                # Preview the adaptive analysis
                st.markdown("### 🔍 Analysis Preview")
                preview_analysis = st.session_state.adaptive_analyzer.analyze_persona_and_task(persona, task)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="dynamic-persona">', unsafe_allow_html=True)
                    st.markdown("**🎯 Detected Focus Areas**")
                    domains = preview_analysis.get("domain_indicators", ["general"])
                    st.write(", ".join(domains).title())
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="dynamic-persona">', unsafe_allow_html=True)
                    st.markdown("**🔑 Key Terms Identified**")
                    keywords = preview_analysis.get("persona_keywords", [])[:3]
                    st.write(", ".join(keywords) if keywords else "General analysis")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="dynamic-persona">', unsafe_allow_html=True)
                    st.markdown("**⚡ Action Focus**")
                    actions = preview_analysis.get("action_verbs", [])[:3]
                    st.write(", ".join(actions) if actions else "Content extraction")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Analyze button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Analyze Complete Collection", type="primary", use_container_width=True):
                        with st.spinner(f"🧠 Analyzing collection of {len(uploaded_files)} documents..."):
                            result = st.session_state.adaptive_analyzer.analyze_collection(
                                uploaded_files,
                                persona,
                                task,
                                collection_name
                            )
                            
                            if result:  # Check if analysis was successful
                                analysis_key = f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                st.session_state.analysis_results[analysis_key] = result
                                st.session_state.current_analysis = analysis_key
                                st.success("✅ Collection analysis completed successfully!")
                                st.rerun()
            
            elif collection_result["collection_type"] == "sample":
                # Handle sample collection
                collection_info = collection_result["collection_info"]
                
                st.info("📝 **Sample Collection Selected**")
                st.write("This demonstrates how the system would process a real collection.")
                st.write(f"**Suggested Persona:** {collection_info['persona']}")
                st.write(f"**Suggested Task:** {collection_info['task']}")
                
                # Option to use suggested values
                if st.button("📋 Use Suggested Persona & Task", type="secondary"):
                    st.session_state.suggested_persona = collection_info['persona']
                    st.session_state.suggested_task = collection_info['task']
                    st.rerun()

def display_adaptive_results():
    """Display adaptive analysis results"""
    if not st.session_state.current_analysis or st.session_state.current_analysis not in st.session_state.analysis_results:
        return
    
    analysis_key = st.session_state.current_analysis
    result = st.session_state.analysis_results[analysis_key]
    metadata = result.get('metadata', {})
    
    st.markdown(f"## 📊 Analysis Results: {metadata.get('collection_name', 'Unknown')}")
    
    # Display adaptation insights
    st.markdown('<div class="task-analysis">', unsafe_allow_html=True)
    st.markdown("### 🧠 Adaptive Analysis Summary")
    
    persona_analysis = metadata.get('persona_analysis', {})
    analysis_insights = metadata.get('analysis_insights', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Persona:** " + metadata.get('persona', 'Unknown'))
        domains = persona_analysis.get('domain_indicators', [])
        if domains:
            st.markdown("**🏢 Domains:** " + ", ".join(domains).title())
    
    with col2:
        st.markdown("**📋 Task Focus:** " + analysis_insights.get('adaptation_summary', 'General'))
        top_factors = analysis_insights.get('top_relevance_factors', [])
        if top_factors:
            st.markdown("**🔑 Top Factors:** " + ", ".join([f[0] for f in top_factors[:3]]))
    
    with col3:
        st.markdown(f"**📄 Documents:** {len(metadata.get('input_documents', []))}")
        st.markdown(f"**🏆 Sections:** {metadata.get('total_sections_extracted', 0)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Adaptive Overview", 
        "🏆 Ranked Sections", 
        "📝 Content Analysis", 
        "📊 Smart Insights",
        "💾 Export Results"
    ])
    
    with tab1:
        display_adaptive_overview_tab(result)
    
    with tab2:
        display_adaptive_sections_tab(result)
    
    with tab3:
        display_adaptive_content_tab(result)
    
    with tab4:
        display_adaptive_insights_tab(result)
    
    with tab5:
        display_adaptive_export_tab(result, metadata.get('collection_name', 'analysis'))

def display_adaptive_overview_tab(result):
    """Display adaptive overview metrics"""
    metadata = result.get('metadata', {})
    persona_analysis = metadata.get('persona_analysis', {})
    analysis_insights = metadata.get('analysis_insights', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Documents", len(metadata.get('input_documents', [])))
    
    with col2:
        st.metric("🏆 Top Sections", metadata.get('total_sections_extracted', 0))
    
    with col3:
        st.metric("📝 Content Items", metadata.get('total_subsections', 0))
    
    with col4:
        domains = persona_analysis.get('domain_indicators', [])
        st.metric("🏢 Domains", len(domains))
    
    # Adaptation details
    st.markdown("### 🧠 How the Tool Adapted to Your Needs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Persona Analysis**")
        st.write(f"**Role:** {metadata.get('persona', 'Unknown')}")
        
        persona_keywords = persona_analysis.get('persona_keywords', [])
        if persona_keywords:
            st.write(f"**Key Terms:** {', '.join(persona_keywords[:8])}")
        
        domains = persona_analysis.get('domain_indicators', [])
        if domains:
            st.write(f"**Identified Domains:** {', '.join(domains).title()}")
    
    with col2:
        st.markdown("**🎯 Task Analysis**")
        st.write(f"**Task:** {metadata.get('job_to_be_done', 'Unknown')}")
        
        action_verbs = persona_analysis.get('action_verbs', [])
        if action_verbs:
            st.write(f"**Action Focus:** {', '.join(action_verbs)}")
        
        context_clues = persona_analysis.get('context_clues', [])
        if context_clues:
            st.write(f"**Context:** {', '.join(context_clues[:5])}")
    
    # Relevance factors analysis
    st.markdown("### 📊 What Made Content Relevant")
    
    top_factors = analysis_insights.get('top_relevance_factors', [])
    if top_factors:
        factor_names = [f[0].replace('_', ' ').title() for f in top_factors]
        factor_counts = [f[1] for f in top_factors]
        
        fig = px.bar(
            x=factor_counts,
            y=factor_names,
            orientation='h',
            title="Top Relevance Factors",
            labels={'x': 'Frequency', 'y': 'Relevance Factor'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Document coverage
    st.markdown("### 📄 Document Coverage Analysis")
    
    doc_coverage = analysis_insights.get('document_coverage', {})
    if doc_coverage:
        fig = px.pie(
            values=list(doc_coverage.values()),
            names=list(doc_coverage.keys()),
            title="Sections Extracted per Document"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_adaptive_sections_tab(result):
    """Display adaptively ranked sections"""
    sections = result.get('extracted_sections', [])
    
    if not sections:
        st.warning("No sections were extracted from the documents.")
        return
    
    st.markdown(f"### 🏆 Top {len(sections)} Most Relevant Sections (Adaptively Ranked)")
    
    # Create enhanced DataFrame
    sections_data = []
    for section in sections:
        relevance_factors = section.get('relevance_factors', [])
        sections_data.append({
            'Rank': section.get('importance_rank', 0),
            'Document': section.get('document', ''),
            'Section Title': section.get('section_title', ''),
            'Page': section.get('page_number', 0),
            'Relevance Factors': ', '.join(relevance_factors) if relevance_factors else 'General'
        })
    
    df_sections = pd.DataFrame(sections_data)
    
    # Filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unique_docs = df_sections['Document'].unique()
        selected_docs = st.multiselect(
            "Filter by Document:",
            unique_docs,
            default=unique_docs
        )
    
    with col2:
        max_rank = df_sections['Rank'].max()
        rank_range = st.slider(
            "Rank Range:",
            1, max_rank,
            (1, min(20, max_rank))
        )
    
    with col3:
        unique_factors = set()
        for factors in df_sections['Relevance Factors']:
            unique_factors.update(factors.split(', '))
        
        selected_factors = st.multiselect(
            "Filter by Relevance Factor:",
            list(unique_factors),
            help="Filter sections by what made them relevant"
        )
    
    # Apply filters
    filtered_df = df_sections[
        (df_sections['Document'].isin(selected_docs)) &
        (df_sections['Rank'] >= rank_range[0]) &
        (df_sections['Rank'] <= rank_range[1])
    ]
    
    if selected_factors:
        filtered_df = filtered_df[
            filtered_df['Relevance Factors'].apply(
                lambda x: any(factor in x for factor in selected_factors)
            )
        ]
    
    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    # Section details
    if not filtered_df.empty:
        st.markdown("### 🔍 Section Analysis")
        
        selected_rank = st.selectbox(
            "Select a section for detailed analysis:",
            filtered_df['Rank'].tolist(),
            format_func=lambda x: f"Rank {x}: {filtered_df[filtered_df['Rank']==x]['Section Title'].iloc[0][:60]}..."
        )
        
        if selected_rank:
            selected_section = next(s for s in sections if s['importance_rank'] == selected_rank)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Document:** {selected_section['document']}")
            with col2:
                st.write(f"**Page:** {selected_section['page_number']}")
            with col3:
                st.write(f"**Rank:** {selected_section['importance_rank']}")
            with col4:
                factors = selected_section.get('relevance_factors', [])
                st.write(f"**Factors:** {len(factors)}")
            
            st.markdown("**Section Title:**")
            st.write(selected_section['section_title'])
            
            if factors:
                st.markdown("**Why This Section Was Relevant:**")
                for factor in factors:
                    st.write(f"• {factor.replace('_', ' ').title()}")

def display_adaptive_content_tab(result):
    """Display adaptive content analysis"""
    subsections = result.get('subsection_analysis', [])
    
    if not subsections:
        st.warning("No content analysis available.")
        return
    
    st.markdown(f"### 📝 Adaptive Content Analysis ({len(subsections)} items)")
    
    # Relevance category distribution
    relevance_categories = [item.get('relevance_category', 'unknown') for item in subsections]
    category_counts = Counter(relevance_categories)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔥 High Relevance", category_counts.get('high', 0))
    with col2:
        st.metric("⚡ Medium Relevance", category_counts.get('medium', 0))
    with col3:
        st.metric("💡 Low Relevance", category_counts.get('low', 0))
    
    # Filter by relevance category
    selected_category = st.selectbox(
        "Filter by Relevance Category:",
        ["All", "High", "Medium", "Low"],
        help="Filter content by relevance level"
    )
    
    filtered_subsections = subsections
    if selected_category != "All":
        filtered_subsections = [
            item for item in subsections 
            if item.get('relevance_category', '').lower() == selected_category.lower()
        ]
    
    # Group by document
    doc_content = defaultdict(list)
    for subsection in filtered_subsections:
        doc_content[subsection['document']].append(subsection)
    
    # Display content by document
    for doc_name, content_list in doc_content.items():
        with st.expander(f"📄 {doc_name} ({len(content_list)} items)"):
            
            # Sort by relevance score
            content_list.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            for i, content in enumerate(content_list[:8]):  # Show top 8 per document
                relevance_score = content.get('relevance_score', 0)
                relevance_category = content.get('relevance_category', 'unknown')
                
                # Color code by relevance
                if relevance_category == 'high':
                    st.markdown(f'<div class="relevance-high">', unsafe_allow_html=True)
                elif relevance_category == 'medium':
                    st.markdown(f'<div class="relevance-medium">', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="relevance-low">', unsafe_allow_html=True)
                
                st.markdown(f"**Item {i+1} (Page {content['page_number']}) - {relevance_category.title()} Relevance**")
                st.progress(relevance_score, text=f"Relevance Score: {relevance_score:.3f}")
                
                text = content['refined_text']
                display_text = text[:400] + "..." if len(text) > 400 else text
                st.write(display_text)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

def display_adaptive_insights_tab(result):
    """Display smart insights from adaptive analysis"""
    metadata = result.get('metadata', {})
    analysis_insights = metadata.get('analysis_insights', {})
    persona_analysis = metadata.get('persona_analysis', {})
    
    st.markdown("### 🧠 Smart Analysis Insights")
    
    # Content distribution analysis
    content_dist = analysis_insights.get('content_distribution', {})
    if content_dist:
        st.markdown("#### 📊 Content Relevance Distribution")
        
        categories = list(content_dist.keys())
        values = list(content_dist.values())
        
        fig = px.bar(
            x=categories,
            y=values,
            title="Content Distribution by Relevance Level",
            labels={'x': 'Relevance Level', 'y': 'Number of Items'},
            color=values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Adaptation effectiveness
    st.markdown("#### 🎯 Adaptation Effectiveness")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Analysis Adaptation**")
        domains = persona_analysis.get('domain_indicators', [])
        if domains:
            st.success(f"✅ Successfully adapted to {', '.join(domains)} domain(s)")
        else:
            st.info("ℹ️ Applied general-purpose analysis")
        
        persona_keywords = persona_analysis.get('persona_keywords', [])
        if persona_keywords:
            st.write(f"**Persona Keywords Used:** {len(persona_keywords)}")
        
        task_keywords = persona_analysis.get('task_keywords', [])
        if task_keywords:
            st.write(f"**Task Keywords Used:** {len(task_keywords)}")
    
    with col2:
        st.markdown("**📈 Relevance Factors Impact**")
        top_factors = analysis_insights.get('top_relevance_factors', [])
        if top_factors:
            for factor, count in top_factors[:5]:
                factor_name = factor.replace('_', ' ').title()
                st.write(f"• **{factor_name}:** {count} matches")
        else:
            st.write("No specific relevance factors identified")
    
    # Document analysis insights
    st.markdown("#### 📄 Document Analysis Insights")
    
    doc_coverage = analysis_insights.get('document_coverage', {})
    if doc_coverage:
        total_sections = sum(doc_coverage.values())
        
        insights_text = f"""
        **Analysis Summary:**
        - Processed {len(doc_coverage)} documents
        - Extracted {total_sections} relevant sections
        - Average sections per document: {total_sections/len(doc_coverage):.1f}
        """
        
        st.markdown(insights_text)
        
        # Document contribution analysis
        fig = px.treemap(
            names=list(doc_coverage.keys()),
            values=list(doc_coverage.values()),
            title="Document Contribution to Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("#### 💡 Recommendations")
    
    recommendations = []
    
    # Based on content distribution
    if content_dist.get('high_relevance', 0) < content_dist.get('low_relevance', 0):
        recommendations.append("Consider refining your persona or task description for better relevance")
    
    # Based on document coverage
    if doc_coverage:
        min_sections = min(doc_coverage.values())
        max_sections = max(doc_coverage.values())
        if max_sections > min_sections * 3:
            recommendations.append("Some documents contributed significantly more content - consider document relevance")
    
    # Based on adaptation
    if not persona_analysis.get('domain_indicators'):
        recommendations.append("Try using more specific domain terminology in your persona/task for better adaptation")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.info(f"{i}. {rec}")
    else:
        st.success("✅ Analysis appears well-optimized for your persona and task!")

def display_adaptive_export_tab(result):
    """Display adaptive export options"""
    st.markdown("### 💾 Export Adaptive Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Complete Analysis (JSON)", use_container_width=True):
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"{collection_name.lower().replace(' ', '_')}_adaptive_analysis.json",
                mime="application/json"
            )
    
    with col2:
        sections = result.get('extracted_sections', [])
        if sections and st.button("📊 Export Sections (CSV)", use_container_width=True):
            # Enhanced CSV with relevance factors
            enhanced_sections = []
            for section in sections:
                enhanced_sections.append({
                    'rank': section.get('importance_rank', 0),
                    'document': section.get('document', ''),
                    'section_title': section.get('section_title', ''),
                    'page_number': section.get('page_number', 0),
                    'relevance_factors': ', '.join(section.get('relevance_factors', []))
                })
            
            df = pd.DataFrame(enhanced_sections)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Enhanced CSV",
                data=csv_data,
                file_name=f"{collection_name.lower().replace(' ', '_')}_sections.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📋 Export Adaptive Report", use_container_width=True):
            report = generate_adaptive_report(result)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"{collection_name.lower().replace(' ', '_')}_adaptive_report.md",
                mime="text/markdown"
            )
    
    # Export preview
    st.markdown("### 👀 Export Preview")
    
    export_type = st.selectbox(
        "Select data to preview:",
        ["Complete Analysis (JSON)", "Enhanced Sections (CSV)", "Adaptive Report (Markdown)"]
    )
    
    if export_type == "Complete Analysis (JSON)":
        st.json(result)
    elif export_type == "Enhanced Sections (CSV)":
        sections = result.get('extracted_sections', [])
        if sections:
            enhanced_sections = []
            for section in sections:
                enhanced_sections.append({
                    'rank': section.get('importance_rank', 0),
                    'document': section.get('document', ''),
                    'section_title': section.get('section_title', ''),
                    'page_number': section.get('page_number', 0),
                    'relevance_factors': ', '.join(section.get('relevance_factors', []))
                })
            df = pd.DataFrame(enhanced_sections)
            st.dataframe(df)
    elif export_type == "Adaptive Report (Markdown)":
        report = generate_adaptive_report(result)
        st.markdown(report)

def generate_adaptive_report(result):
    """Generate an adaptive analysis report"""
    metadata = result.get('metadata', {})
    persona_analysis = metadata.get('persona_analysis', {})
    analysis_insights = metadata.get('analysis_insights', {})
    sections = result.get('extracted_sections', [])
    
    report = f"""# {collection_name} - Adaptive Analysis Report

## 🧠 Adaptive Analysis Overview

### Persona & Task Analysis
- **Persona:** {metadata.get('persona', 'Unknown')}
- **Task:** {metadata.get('job_to_be_done', 'Unknown')}
- **Analysis Date:** {metadata.get('processing_timestamp', 'Unknown')}

### Adaptation Details
- **Identified Domains:** {', '.join(persona_analysis.get('domain_indicators', ['General'])).title()}
- **Persona Keywords:** {len(persona_analysis.get('persona_keywords', []))} identified
- **Task Keywords:** {len(persona_analysis.get('task_keywords', []))} identified
- **Action Verbs:** {', '.join(persona_analysis.get('action_verbs', []))}

## 📊 Analysis Results

### Document Processing
- **Documents Processed:** {len(metadata.get('input_documents', []))}
- **Total Sections Extracted:** {metadata.get('total_sections_extracted', 0)}
- **Total Content Items:** {metadata.get('total_subsections', 0)}

### Input Documents
"""
    
    for doc in metadata.get('input_documents', []):
        report += f"- {doc}\n"
    
    # Top relevance factors
    top_factors = analysis_insights.get('top_relevance_factors', [])
    if top_factors:
        report += "\n### 🎯 Top Relevance Factors\n"
        for factor, count in top_factors:
            factor_name = factor.replace('_', ' ').title()
            report += f"- **{factor_name}:** {count} matches\n"
    
    # Content distribution
    content_dist = analysis_insights.get('content_distribution', {})
    if content_dist:
        report += "\n### 📈 Content Relevance Distribution\n"
        for category, count in content_dist.items():
            category_name = category.replace('_', ' ').title()
            report += f"- **{category_name}:** {count} items\n"
    
    # Top sections
    report += f"\n## 🏆 Top {min(15, len(sections))} Most Relevant Sections\n\n"
    
    for i, section in enumerate(sections[:15], 1):
        report += f"{i}. **{section.get('section_title', 'Unknown')}**\n"
        report += f"   - Document: {section.get('document', 'Unknown')}\n"
        report += f"   - Page: {section.get('page_number', 'Unknown')}\n"
        report += f"   - Rank: {section.get('importance_rank', 'Unknown')}\n"
        
        factors = section.get('relevance_factors', [])
        if factors:
            report += f"   - Relevance Factors: {', '.join(factors)}\n"
        report += "\n"
    
    # Adaptation summary
    report += f"\n## 🎯 Adaptation Summary\n\n"
    report += f"{analysis_insights.get('adaptation_summary', 'General analysis performed')}\n\n"
    
    report += "### Key Adaptations Made:\n"
    domains = persona_analysis.get('domain_indicators', [])
    if domains:
        report += f"- Applied {', '.join(domains)} domain-specific analysis\n"
    
    persona_keywords = persona_analysis.get('persona_keywords', [])
    if persona_keywords:
        report += f"- Used {len(persona_keywords)} persona-specific keywords\n"
    
    task_keywords = persona_analysis.get('task_keywords', [])
    if task_keywords:
        report += f"- Incorporated {len(task_keywords)} task-specific terms\n"
    
    action_verbs = persona_analysis.get('action_verbs', [])
    if action_verbs:
        report += f"- Focused on action-oriented content: {', '.join(action_verbs)}\n"
    
    report += f"\n---\n*Adaptive analysis report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    report += "*This analysis was dynamically adapted to your specific persona and task requirements.*"
    
    return report

def main():
    """Main application function"""
    initialize_session_state()
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Adaptive Smart PDF Tool")
        st.markdown("**Adapts to ANY persona and task!**")
        
        st.markdown("### ✨ Adaptive Features")
        st.markdown("""
        - 🎯 **Dynamic Persona Analysis**
        - 🧠 **Intelligent Task Understanding**
        - 🔍 **Domain-Specific Adaptation**
        - 📊 **Relevance Factor Analysis**
        - 💡 **Smart Content Refinement**
        - 📈 **Adaptive Ranking Algorithm**
        """)
        
        st.markdown("### 🚀 How It Works")
        st.markdown("""
        1. **Analyze** your persona and task
        2. **Adapt** algorithms dynamically
        3. **Extract** most relevant content
        4. **Rank** by adaptive importance
        5. **Provide** tailored insights
        """)
        
        # Example use cases
        with st.expander("💡 Example Use Cases"):
            st.markdown("""
            **Research Analyst** analyzing market reports
            **Compliance Officer** reviewing regulations
            **Product Manager** extracting feature requirements
            **Consultant** finding best practices
            **Investor** analyzing financial documents
            **Academic** reviewing literature
            """)
        
        # Reset button
        if st.button("🔄 New Analysis", type="secondary"):
            st.session_state.analysis_results = {}
            st.session_state.current_analysis = None
            st.rerun()
    
    # Main content
    if not st.session_state.current_analysis:
        display_adaptive_setup()
    else:
        display_adaptive_results()
        
        # Option to start new analysis
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧠 Start New Adaptive Analysis", type="secondary", use_container_width=True):
                st.session_state.current_analysis = None
                st.rerun()

if __name__ == "__main__":
    main()
