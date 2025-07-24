"""
Main Streamlit application for the Adobe PDF Challenge
"""

import streamlit as st
import pandas as pd
from pdf_processor import PDFProcessor
from visualizer import PDFVisualizer
import json
import time

# Page configuration
st.set_page_config(
    page_title="Adobe PDF Intelligence Challenge",
    page_icon="📄",
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
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF0000;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'processor' not in st.session_state:
        st.session_state.processor = PDFProcessor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = PDFVisualizer()

def display_header():
    """Display the main header"""
    st.markdown('<div class="main-header">📄 Adobe PDF Intelligence Challenge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Connecting the Dots - Rethink Reading, Rediscover Knowledge</div>', unsafe_allow_html=True)

def display_file_upload():
    """Display file upload section"""
    st.markdown('<div class="section-header">📤 Upload Your PDF</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze its structure and extract insights"
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze PDF", type="primary", use_container_width=True):
                with st.spinner("Processing PDF... This may take a few moments."):
                    progress_bar = st.progress(0)
                    
                    # Simulate progress updates
                    progress_bar.progress(20)
                    time.sleep(0.5)
                    
                    # Process the PDF
                    result = st.session_state.processor.process_pdf(uploaded_file)
                    progress_bar.progress(100)
                    
                    if 'error' in result:
                        st.error(f"Error processing PDF: {result['error']}")
                    else:
                        st.session_state.processed_data = result
                        st.success("✅ PDF processed successfully!")
                        st.rerun()

def display_overview(data):
    """Display overview metrics"""
    st.markdown('<div class="section-header">📊 Document Overview</div>', unsafe_allow_html=True)
    
    metadata = data.get('metadata', {})
    insights = data.get('insights', {})
    statistics = insights.get('statistics', {})
    
    # Display metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Document Information**")
        st.write(f"**Title:** {metadata.get('title', 'Unknown')}")
        st.write(f"**Author:** {metadata.get('author', 'Unknown')}")
        st.write(f"**Total Pages:** {metadata.get('total_pages', 0)}")
    
    with col2:
        st.markdown("**📈 Content Statistics**")
        st.write(f"**Total Words:** {statistics.get('total_words', 0):,}")
        st.write(f"**Total Sentences:** {statistics.get('total_sentences', 0):,}")
        st.write(f"**Total Sections:** {statistics.get('total_sections', 0)}")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📖 Reading Time",
            value=f"{statistics.get('reading_time_minutes', 0):.1f} min"
        )
    
    with col2:
        readability = insights.get('readability', {})
        st.metric(
            label="📚 Reading Level",
            value=f"Grade {readability.get('flesch_kincaid_grade', 0):.1f}"
        )
    
    with col3:
        st.metric(
            label="📄 Avg Words/Section",
            value=f"{statistics.get('avg_words_per_section', 0):.0f}"
        )
    
    with col4:
        st.metric(
            label="🎯 Readability Score",
            value=f"{readability.get('flesch_reading_ease', 0):.1f}"
        )

def display_structure_analysis(data):
    """Display document structure analysis"""
    st.markdown('<div class="section-header">🏗️ Document Structure</div>', unsafe_allow_html=True)
    
    structure = data.get('structure', {})
    headings = structure.get('headings', [])
    sections = structure.get('sections', [])
    
    if headings:
        # Display structure tree
        fig_tree = st.session_state.visualizer.create_structure_tree(headings)
        if fig_tree:
            st.plotly_chart(fig_tree, use_container_width=True)
        
        # Display table of contents
        st.markdown("**📑 Table of Contents**")
        toc_data = []
        for heading in headings:
            toc_data.append({
                'Level': heading.get('level', 1),
                'Title': heading.get('title', ''),
                'Page': heading.get('page', 1)
            })
        
        if toc_data:
            df_toc = pd.DataFrame(toc_data)
            st.dataframe(df_toc, use_container_width=True)
    
    # Display sections summary
    if sections:
        st.markdown("**📝 Sections Summary**")
        sections_data = []
        for section in sections:
            sections_data.append({
                'Section': section.get('title', '')[:50] + '...' if len(section.get('title', '')) > 50 else section.get('title', ''),
                'Page': section.get('page', 1),
                'Word Count': section.get('word_count', 0),
                'Content Preview': section.get('content', '')[:100] + '...' if len(section.get('content', '')) > 100 else section.get('content', '')
            })
        
        if sections_data:
            df_sections = pd.DataFrame(sections_data)
            st.dataframe(df_sections, use_container_width=True)

def display_content_relationships(data):
    """Display content relationship analysis"""
    st.markdown('<div class="section-header">🔗 Content Relationships</div>', unsafe_allow_html=True)
    
    relationships = data.get('relationships', {})
    
    # Display similarity heatmap
    similarity_matrix = relationships.get('similarity_matrix', [])
    section_titles = relationships.get('section_titles', [])
    
    if similarity_matrix and section_titles:
        fig_heatmap = st.session_state.visualizer.create_similarity_heatmap(similarity_matrix, section_titles)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Display content clusters
    sections = data.get('structure', {}).get('sections', [])
    clusters = relationships.get('clusters', [])
    
    if sections and clusters:
        fig_clusters = st.session_state.visualizer.create_content_clusters(sections, clusters)
        if fig_clusters:
            st.plotly_chart(fig_clusters, use_container_width=True)
    
    # Display relationship network
    fig_network = st.session_state.visualizer.create_relationship_network(relationships)
    if fig_network:
        st.plotly_chart(fig_network, use_container_width=True)
    
    # Display related sections
    related_sections = relationships.get('related_sections', [])
    if related_sections:
        st.markdown("**🔍 Section Relationships**")
        
        for section_data in related_sections[:5]:  # Show top 5
            section_title = section_data.get('section', '')
            related = section_data.get('related', [])
            
            if related:
                with st.expander(f"📄 {section_title[:60]}..."):
                    st.write("**Related Sections:**")
                    for rel in related:
                        similarity = rel.get('similarity', 0)
                        rel_title = rel.get('title', '')
                        st.write(f"• {rel_title} (Similarity: {similarity:.3f})")

def display_insights_analysis(data):
    """Display key insights and analysis"""
    st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
    
    insights = data.get('insights', {})
    
    # Display readability analysis
    readability = insights.get('readability', {})
    if readability:
        fig_readability = st.session_state.visualizer.create_readability_chart(readability)
        if fig_readability:
            st.plotly_chart(fig_readability, use_container_width=True)
    
    # Display statistics overview
    statistics = insights.get('statistics', {})
    if statistics:
        fig_stats = st.session_state.visualizer.create_statistics_overview(statistics)
        if fig_stats:
            st.plotly_chart(fig_stats, use_container_width=True)
    
    # Display key entities
    entities = insights.get('key_entities', [])
    if entities:
        st.markdown("**🏷️ Key Entities Found**")
        
        cols = st.columns(min(3, len(entities)))
        for i, entity_group in enumerate(entities[:3]):
            with cols[i]:
                category = entity_group.get('category', 'Unknown')
                entity_list = entity_group.get('entities', [])
                
                st.markdown(f"**{category}**")
                for entity in entity_list[:5]:  # Show top 5 entities
                    st.write(f"• {entity}")
    
    # Display summary points
    summary_points = insights.get('summary_points', [])
    if summary_points:
        st.markdown("**📝 Key Summary Points**")
        for i, point in enumerate(summary_points, 1):
            st.write(f"{i}. {point}")
    
    # Display word cloud
    full_text = ' '.join([page['text'] for page in data.get('text_content', [])])
    if full_text:
        wordcloud_img = st.session_state.visualizer.create_wordcloud(full_text)
        if wordcloud_img:
            st.markdown("**☁️ Word Cloud**")
            st.image(wordcloud_img, use_column_width=True)

def display_export_options(data):
    """Display export options"""
    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        if st.button("📄 Export as JSON", use_container_width=True):
            json_data = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="pdf_analysis.json",
                mime="application/json"
            )
    
    with col2:
        # Export structure as CSV
        structure = data.get('structure', {})
        sections = structure.get('sections', [])
        if sections:
            if st.button("📊 Export Structure as CSV", use_container_width=True):
                df = pd.DataFrame(sections)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="document_structure.csv",
                    mime="text/csv"
                )
    
    with col3:
        # Export insights summary
        if st.button("📋 Export Summary Report", use_container_width=True):
            insights = data.get('insights', {})
            metadata = data.get('metadata', {})
            
            report = f"""
# PDF Analysis Report

## Document Information
- Title: {metadata.get('title', 'Unknown')}
- Author: {metadata.get('author', 'Unknown')}
- Pages: {metadata.get('total_pages', 0)}

## Statistics
- Total Words: {insights.get('statistics', {}).get('total_words', 0):,}
- Total Sentences: {insights.get('statistics', {}).get('total_sentences', 0):,}
- Total Sections: {insights.get('statistics', {}).get('total_sections', 0)}
- Reading Time: {insights.get('statistics', {}).get('reading_time_minutes', 0):.1f} minutes

## Readability
- Flesch Reading Ease: {insights.get('readability', {}).get('flesch_reading_ease', 0):.1f}
- Grade Level: {insights.get('readability', {}).get('flesch_kincaid_grade', 0):.1f}

## Key Summary Points
"""
            
            summary_points = insights.get('summary_points', [])
            for i, point in enumerate(summary_points, 1):
                report += f"{i}. {point}\n"
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="pdf_analysis_report.md",
                mime="text/markdown"
            )

def main():
    """Main application function"""
    initialize_session_state()
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Challenge 1a")
        st.markdown("**Objective:** Extract structured outlines from PDFs and understand content relationships")
        
        st.markdown("## 🚀 Features")
        st.markdown("""
        - 📄 PDF text extraction
        - 🏗️ Structure analysis
        - 🔗 Content relationships
        - 💡 Key insights
        - 📊 Interactive visualizations
        - 💾 Export capabilities
        """)
        
        st.markdown("## 📖 How to Use")
        st.markdown("""
        1. Upload a PDF file
        2. Click 'Analyze PDF'
        3. Explore the results
        4. Export your findings
        """)
    
    # Main content
    if st.session_state.processed_data is None:
        display_file_upload()
        
        # Show sample information
        st.markdown("---")
        st.markdown("## 🎯 What This Tool Does")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📄 Structure Extraction**
            - Identifies headings and sections
            - Creates document hierarchy
            - Generates table of contents
            """)
        
        with col2:
            st.markdown("""
            **🔗 Content Analysis**
            - Finds related sections
            - Calculates content similarity
            - Creates relationship networks
            """)
        
        with col3:
            st.markdown("""
            **💡 Smart Insights**
            - Extracts key entities
            - Analyzes readability
            - Generates summaries
            """)
    
    else:
        # Display analysis results
        data = st.session_state.processed_data
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🏗️ Structure", 
            "🔗 Relationships", 
            "💡 Insights", 
            "💾 Export"
        ])
        
        with tab1:
            display_overview(data)
        
        with tab2:
            display_structure_analysis(data)
        
        with tab3:
            display_content_relationships(data)
        
        with tab4:
            display_insights_analysis(data)
        
        with tab5:
            display_export_options(data)
        
        # Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Analyze New PDF", type="secondary", use_container_width=True):
                st.session_state.processed_data = None
                st.rerun()

if __name__ == "__main__":
    main()
