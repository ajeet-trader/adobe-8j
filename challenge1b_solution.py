"""
Adobe India Hackathon 2025 - Challenge 1B: Multi-Collection PDF Analysis
Complete end-to-end solution for persona-based content analysis
"""

import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaBasedAnalyzer:
    def __init__(self):
        """Initialize the persona-based PDF analyzer with enhanced keyword mappings"""
        
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
    
    def analyze_collection(self, input_config_path: str) -> Dict[str, Any]:
        """
        Main method to analyze a collection based on input configuration
        
        Args:
            input_config_path: Path to the input JSON configuration file
            
        Returns:
            Dictionary containing analysis results in required format
        """
        logger.info(f"Starting analysis for: {input_config_path}")
        
        try:
            with open(input_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return self._create_error_response(str(e))
        
        persona = config["persona"]["role"]
        task = config["job_to_be_done"]["task"]
        documents = config["documents"]
        
        logger.info(f"Processing {len(documents)} documents for persona: {persona}")
        logger.info(f"Task: {task}")
        
        # Process each PDF in the collection
        all_extracted_sections = []
        all_subsection_analysis = []
        
        collection_path = Path(input_config_path).parent
        pdfs_path = collection_path / "PDFs"
        
        for doc_info in documents:
            pdf_path = pdfs_path / doc_info["filename"]
            
            if pdf_path.exists():
                logger.info(f"Processing: {doc_info['filename']}")
                sections, subsections = self._process_pdf(pdf_path, persona, task, doc_info["title"])
                all_extracted_sections.extend(sections)
                all_subsection_analysis.extend(subsections)
            else:
                logger.warning(f"PDF not found: {pdf_path}")
        
        # Rank sections by importance
        ranked_sections = self._rank_by_importance(all_extracted_sections, persona, task)
        
        # Ensure diverse representation across documents
        balanced_sections = self._balance_document_representation(ranked_sections)
        
        # Create final response
        response = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in documents],
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_extracted": len(balanced_sections),
                "total_subsections": len(all_subsection_analysis)
            },
            "extracted_sections": balanced_sections[:50],  # Limit to top 50 most relevant
            "subsection_analysis": all_subsection_analysis[:100]  # Limit to top 100 subsections
        }
        
        logger.info(f"Analysis complete. Extracted {len(balanced_sections)} sections and {len(all_subsection_analysis)} subsections")
        return response
    
    def _process_pdf(self, pdf_path: Path, persona: str, task: str, doc_title: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract and analyze content from a single PDF
        
        Args:
            pdf_path: Path to the PDF file
            persona: User persona role
            task: Specific task description
            doc_title: Title of the document
            
        Returns:
            Tuple of (extracted_sections, subsection_analysis)
        """
        sections = []
        subsections = []
        
        try:
            doc = fitz.open(pdf_path)
            
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
                                        "document": pdf_path.name,
                                        "section_title": self._clean_section_title(block_text),
                                        "importance_rank": relevance_score,
                                        "page_number": page_num + 1,
                                        "document_title": doc_title
                                    })
                                
                                # Add to subsection analysis
                                refined_text = self._refine_text_for_persona(block_text, persona, task)
                                if refined_text:
                                    subsections.append({
                                        "document": pdf_path.name,
                                        "refined_text": refined_text,
                                        "page_number": page_num + 1,
                                        "relevance_score": relevance_score
                                    })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        
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
        """Check if text is meaningful (not just whitespace, page numbers, etc.)"""
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
        # Remove extra whitespace and clean up
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Limit length for section titles
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
        
        return cleaned
    
    def _calculate_relevance(self, text: str, persona: str, task: str) -> float:
        """
        Calculate relevance score for text based on persona and task
        
        Args:
            text: Text content to analyze
            persona: User persona
            task: Specific task
            
        Returns:
            Relevance score between 0 and 1
        """
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
        task_matches = 0
        
        for task_phrase, keywords in self.task_keywords.items():
            if task_phrase.lower() in task.lower():
                task_matches = sum(1 for keyword in keywords if keyword in text_lower)
                task_score = task_matches / len(keywords)
                break
        
        # Combine scores with weights
        final_score = (persona_score * 0.7) + (task_score * 0.3)
        
        # Boost score for longer, more substantial content
        length_boost = min(len(text) / 500, 0.2)  # Up to 20% boost for longer text
        final_score += length_boost
        
        return min(final_score, 1.0)
    
    def _refine_text_for_persona(self, text: str, persona: str, task: str) -> str:
        """
        Refine extracted text to be more relevant for the specific persona
        
        Args:
            text: Original text
            persona: User persona
            task: Specific task
            
        Returns:
            Refined text or None if not relevant enough
        """
        # Clean up the text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # For different personas, apply different refinement strategies
        if persona == "Travel Planner":
            return self._refine_for_travel_planner(cleaned_text, task)
        elif persona == "HR Professional":
            return self._refine_for_hr_professional(cleaned_text, task)
        elif persona == "Food Contractor":
            return self._refine_for_food_contractor(cleaned_text, task)
        
        return cleaned_text if len(cleaned_text) > 20 else None
    
    def _refine_for_travel_planner(self, text: str, task: str) -> str:
        """Refine text for travel planner persona"""
        # Look for actionable travel information
        if any(keyword in text.lower() for keyword in ["book", "visit", "go to", "recommended", "must-see", "cost", "price"]):
            return text
        
        # Extract practical information
        if re.search(r'\d+.*(?:euro|€|hour|day|minute)', text.lower()):
            return text
        
        return text if len(text) > 30 else None
    
    def _refine_for_hr_professional(self, text: str, task: str) -> str:
        """Refine text for HR professional persona"""
        # Look for step-by-step instructions
        if re.search(r'\d+\.\s|step \d+|first|then|next|finally', text.lower()):
            return text
        
        # Look for form-related content
        if any(keyword in text.lower() for keyword in ["create", "add", "insert", "field", "button", "signature"]):
            return text
        
        return text if len(text) > 25 else None
    
    def _refine_for_food_contractor(self, text: str, task: str) -> str:
        """Refine text for food contractor persona"""
        # Look for recipes and ingredients
        if re.search(r'\d+\s*(?:cup|tbsp|tsp|oz|lb|gram|kg|ml|liter)', text.lower()):
            return text
        
        # Look for cooking instructions
        if any(keyword in text.lower() for keyword in ["cook", "bake", "fry", "boil", "mix", "add", "serve", "prepare"]):
            return text
        
        return text if len(text) > 20 else None
    
    def _rank_by_importance(self, sections: List[Dict], persona: str, task: str) -> List[Dict]:
        """
        Rank sections by importance for the given persona and task
        
        Args:
            sections: List of extracted sections
            persona: User persona
            task: Specific task
            
        Returns:
            Ranked list of sections
        """
        # Sort by relevance score (importance_rank) in descending order
        sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        # Assign final importance ranking
        for i, section in enumerate(sections, 1):
            section["importance_rank"] = i
        
        return sections
    
    def _balance_document_representation(self, sections: List[Dict]) -> List[Dict]:
        """
        Ensure balanced representation across all documents
        
        Args:
            sections: Ranked sections
            
        Returns:
            Balanced list of sections
        """
        # Group sections by document
        doc_sections = defaultdict(list)
        for section in sections:
            doc_sections[section["document"]].append(section)
        
        # Interleave sections from different documents
        balanced = []
        max_per_doc = max(3, len(sections) // len(doc_sections))  # At least 3 per document
        
        for doc, doc_section_list in doc_sections.items():
            balanced.extend(doc_section_list[:max_per_doc])
        
        # Sort the balanced list by original importance ranking
        balanced.sort(key=lambda x: x["importance_rank"])
        
        # Re-rank after balancing
        for i, section in enumerate(balanced, 1):
            section["importance_rank"] = i
        
        return balanced
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response in case of processing failure"""
        return {
            "metadata": {
                "error": error_message,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

def process_all_collections():
    """Process all three collections"""
    analyzer = PersonaBasedAnalyzer()
    base_path = Path(".")
    
    collections = [
        "Collection 1",
        "Collection 2", 
        "Collection 3"
    ]
    
    results = {}
    
    for collection_name in collections:
        collection_path = base_path / collection_name
        input_file = collection_path / "challenge1b_input.json"
        output_file = collection_path / "challenge1b_output.json"
        
        if input_file.exists():
            logger.info(f"Processing {collection_name}...")
            
            try:
                result = analyzer.analyze_collection(str(input_file))
                
                # Save result to output file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                results[collection_name] = result
                logger.info(f"✅ Successfully processed {collection_name}")
                logger.info(f"   Generated: {output_file}")
                logger.info(f"   Sections: {len(result.get('extracted_sections', []))}")
                logger.info(f"   Subsections: {len(result.get('subsection_analysis', []))}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {collection_name}: {e}")
                results[collection_name] = {"error": str(e)}
        else:
            logger.warning(f"⚠️ Input file not found: {input_file}")
            results[collection_name] = {"error": "Input file not found"}
    
    return results

def main():
    """Main execution function"""
    logger.info("Starting Adobe India Hackathon 2025 - Challenge 1B Solution")
    logger.info("=" * 60)
    
    # Process all collections
    results = process_all_collections()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    
    for collection, result in results.items():
        if "error" in result:
            logger.error(f"{collection}: ❌ {result['error']}")
        else:
            sections_count = len(result.get('extracted_sections', []))
            subsections_count = len(result.get('subsection_analysis', []))
            logger.info(f"{collection}: ✅ {sections_count} sections, {subsections_count} subsections")
    
    logger.info("\nChallenge 1B processing complete!")

if __name__ == "__main__":
    main()
