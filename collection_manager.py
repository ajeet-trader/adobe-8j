"""
Collection Management Module for Challenge 1B
Handles multiple PDF collections and cross-document analysis
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Any
import zipfile
import io

class CollectionManager:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.max_collection_size = 50  # Maximum files per collection
        self.min_collection_size = 2   # Minimum files per collection (Challenge 1B requirement)
    
    def create_collection_interface(self):
        """Create interface for managing PDF collections"""
        st.markdown("## 📚 Collection Management")
        
        # Collection creation options
        collection_method = st.radio(
            "How would you like to create your collection?",
            ["Upload Individual Files", "Upload Zip Archive", "Load Sample Collection"],
            help="Choose the method that works best for your document collection"
        )
        
        if collection_method == "Upload Individual Files":
            return self._individual_file_upload()
        elif collection_method == "Upload Zip Archive":
            return self._zip_upload()
        else:
            return self._sample_collection_loader()
    
    def _individual_file_upload(self):
        """Handle individual file uploads"""
        st.markdown("### 📄 Upload Individual PDF Files")
        
        uploaded_files = st.file_uploader(
            "Select multiple PDF files for your collection",
            type="pdf",
            accept_multiple_files=True,
            help=f"Upload {self.min_collection_size}-{self.max_collection_size} PDF files that form a cohesive collection"
        )
        
        if uploaded_files:
            return self._validate_and_process_collection(uploaded_files)
        
        return None
    
    def _zip_upload(self):
        """Handle zip file upload containing PDFs"""
        st.markdown("### 📦 Upload Zip Archive")
        st.info("Upload a zip file containing multiple PDF documents")
        
        zip_file = st.file_uploader(
            "Select zip file containing PDFs",
            type="zip",
            help="Zip file should contain only PDF files for the collection"
        )
        
        if zip_file:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    pdf_files = []
                    file_list = zip_ref.namelist()
                    
                    for file_name in file_list:
                        if file_name.lower().endswith('.pdf') and not file_name.startswith('__MACOSX'):
                            file_data = zip_ref.read(file_name)
                            
                            # Create a file-like object
                            pdf_file = io.BytesIO(file_data)
                            pdf_file.name = Path(file_name).name
                            pdf_files.append(pdf_file)
                    
                    if pdf_files:
                        st.success(f"✅ Extracted {len(pdf_files)} PDF files from zip archive")
                        return self._validate_and_process_collection(pdf_files)
                    else:
                        st.error("❌ No PDF files found in the zip archive")
                        return None
            
            except Exception as e:
                st.error(f"❌ Error processing zip file: {e}")
                return None
        
        return None
    
    def _sample_collection_loader(self):
        """Load predefined sample collections"""
        st.markdown("### 🎯 Sample Collections")
        st.info("Choose from predefined sample collections to test the system")
        
        sample_collections = {
            "Travel Planning": {
                "description": "Collection of travel guides and planning documents",
                "files": ["Cities_Guide.pdf", "Cuisine_Guide.pdf", "Hotels_Restaurants.pdf", "Activities.pdf", "Transportation.pdf"],
                "persona": "Travel Planner",
                "task": "Plan a 4-day group trip to South of France"
            },
            "HR Onboarding": {
                "description": "Adobe Acrobat tutorials for HR professionals",
                "files": ["Creating_Forms.pdf", "Digital_Signatures.pdf", "Workflow_Automation.pdf", "Document_Security.pdf"],
                "persona": "HR Professional", 
                "task": "Create fillable forms for employee onboarding and compliance"
            },
            "Corporate Catering": {
                "description": "Recipe collections for corporate event catering",
                "files": ["Vegetarian_Recipes.pdf", "Gluten_Free_Options.pdf", "Buffet_Menus.pdf", "Corporate_Catering.pdf"],
                "persona": "Food Contractor",
                "task": "Plan vegetarian buffet menu for corporate gathering"
            },
            "Legal Research": {
                "description": "Legal documents and compliance guides",
                "files": ["Contract_Law.pdf", "Regulatory_Compliance.pdf", "Case_Studies.pdf", "Legal_Procedures.pdf"],
                "persona": "Legal Advisor",
                "task": "Research compliance requirements for new business venture"
            },
            "Financial Analysis": {
                "description": "Financial reports and investment analysis documents",
                "files": ["Market_Analysis.pdf", "Investment_Guide.pdf", "Risk_Assessment.pdf", "Financial_Statements.pdf"],
                "persona": "Financial Analyst",
                "task": "Analyze investment opportunities for portfolio diversification"
            }
        }
        
        selected_collection = st.selectbox(
            "Choose a sample collection:",
            list(sample_collections.keys()),
            help="Select a predefined collection to demonstrate the system capabilities"
        )
        
        if selected_collection:
            collection_info = sample_collections[selected_collection]
            
            # Display collection details
            with st.expander("📋 Collection Details", expanded=True):
                st.write(f"**Description:** {collection_info['description']}")
                st.write(f"**Suggested Persona:** {collection_info['persona']}")
                st.write(f"**Suggested Task:** {collection_info['task']}")
                st.write("**Files in Collection:**")
                for file in collection_info['files']:
                    st.write(f"• {file}")
            
            # Note about sample collections
            st.warning("📝 **Note**: Sample collections are for demonstration purposes. Upload your own PDF files to use the actual analysis functionality.")
            
            return {
                "collection_type": "sample",
                "collection_info": collection_info,
                "files": None
            }
        
        return None
    
    def _validate_and_process_collection(self, uploaded_files):
        """Validate and process the uploaded collection"""
        if len(uploaded_files) < self.min_collection_size:
            st.error(f"❌ Challenge 1B requires at least {self.min_collection_size} PDF files in a collection. You uploaded {len(uploaded_files)} file(s).")
            st.info("💡 **Why multiple files?** Challenge 1B is designed to find connections and insights across multiple related documents, not just analyze a single file.")
            return None
        
        if len(uploaded_files) > self.max_collection_size:
            st.warning(f"⚠️ Collection size limited to {self.max_collection_size} files. Using first {self.max_collection_size} files.")
            uploaded_files = uploaded_files[:self.max_collection_size]
        
        # Display collection validation results
        st.success(f"✅ Valid collection with {len(uploaded_files)} PDF files")
        
        # Collection statistics
        total_size = sum(getattr(file, 'size', 0) for file in uploaded_files)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Documents", len(uploaded_files))
        with col2:
            st.metric("💾 Total Size", f"{total_size/1024/1024:.1f} MB")
        with col3:
            st.metric("📊 Avg Size", f"{total_size/len(uploaded_files)/1024:.0f} KB")
        
        # File details table
        with st.expander("📋 Collection File Details"):
            file_data = []
            for i, file in enumerate(uploaded_files, 1):
                file_data.append({
                    "File #": i,
                    "Document Name": getattr(file, 'name', f'Document_{i}'),
                    "Size (KB)": f"{getattr(file, 'size', 0)/1024:.1f}",
                    "Status": "✅ Ready"
                })
            
            import pandas as pd
            df_files = pd.DataFrame(file_data)
            st.dataframe(df_files, use_container_width=True, hide_index=True)
        
        return {
            "collection_type": "uploaded",
            "files": uploaded_files,
            "collection_stats": {
                "file_count": len(uploaded_files),
                "total_size": total_size,
                "avg_size": total_size / len(uploaded_files)
            }
        }
    
    def generate_collection_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a comprehensive collection analysis report"""
        metadata = analysis_result.get('metadata', {})
        
        report = f"""# Collection Analysis Report

## 📚 Collection Overview
- **Collection Name:** {metadata.get('collection_name', 'Unknown')}
- **Collection Size:** {metadata.get('collection_size', 0)} documents
- **Analysis Date:** {metadata.get('processing_timestamp', 'Unknown')}

## 🎯 Analysis Configuration
- **Persona:** {metadata.get('persona', 'Unknown')}
- **Task:** {metadata.get('job_to_be_done', 'Unknown')}

## 📊 Processing Results
- **Total Sections Extracted:** {metadata.get('total_sections_extracted', 0)}
- **Total Content Items:** {metadata.get('total_subsections', 0)}

## 📄 Documents in Collection
"""
        
        document_summaries = metadata.get('document_summaries', {})
        for doc_name, summary in document_summaries.items():
            report += f"""
### {doc_name}
- Sections Found: {summary.get('sections_found', 0)}
- Content Items: {summary.get('content_items', 0)}
- Average Relevance: {summary.get('avg_relevance', 0):.3f}
"""
        
        # Cross-document insights
        cross_insights = metadata.get('cross_document_insights', {})
        if cross_insights:
            report += "\n## 🔗 Cross-Document Analysis\n"
            
            document_overlap = cross_insights.get('document_overlap', {})
            if document_overlap:
                report += "\n### Document Topic Overlap\n"
                for docs, topics in document_overlap.items():
                    report += f"- **{docs}**: {', '.join(topics)}\n"
        
        # Top sections
        sections = analysis_result.get('extracted_sections', [])
        if sections:
            report += f"\n## 🏆 Top {min(10, len(sections))} Most Relevant Sections\n"
            for i, section in enumerate(sections[:10], 1):
                report += f"""
{i}. **{section.get('section_title', 'Unknown')}**
   - Document: {section.get('document', 'Unknown')}
   - Page: {section.get('page_number', 'Unknown')}
   - Relevance Rank: {section.get('importance_rank', 'Unknown')}
"""
        
        report += f"\n---\n*Collection analysis completed on {metadata.get('processing_timestamp', 'Unknown')}*"
        
        return report

# Integration with main app
def integrate_collection_manager():
    """Integrate collection manager with the main application"""
    if 'collection_manager' not in st.session_state:
        st.session_state.collection_manager = CollectionManager()
    
    return st.session_state.collection_manager
