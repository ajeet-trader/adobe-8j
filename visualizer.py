"""
Visualization module for creating interactive charts and graphs
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

class PDFVisualizer:
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_structure_tree(self, headings):
        """Create a hierarchical tree visualization of document structure"""
        if not headings:
            return None
        
        fig = go.Figure()
        
        # Create tree structure
        levels = {}
        for heading in headings:
            level = heading.get('level', 1)
            if level not in levels:
                levels[level] = []
            levels[level].append(heading)
        
        # Plot tree nodes
        y_positions = {}
        y_counter = 0
        
        for level in sorted(levels.keys()):
            x_pos = level * 2
            for heading in levels[level]:
                fig.add_trace(go.Scatter(
                    x=[x_pos],
                    y=[y_counter],
                    mode='markers+text',
                    text=[heading['title'][:30] + '...' if len(heading['title']) > 30 else heading['title']],
                    textposition='middle right',
                    marker=dict(
                        size=20 - level * 2,
                        color=self.colors[level % len(self.colors)],
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>{heading['title']}</b><br>Page: {heading['page']}<br>Level: {heading['level']}<extra></extra>"
                ))
                y_positions[heading['title']] = (x_pos, y_counter)
                y_counter += 1
        
        fig.update_layout(
            title="Document Structure Tree",
            xaxis_title="Heading Level",
            yaxis_title="Section Order",
            height=max(400, len(headings) * 30),
            showlegend=False
        )
        
        return fig
    
    def create_similarity_heatmap(self, similarity_matrix, section_titles):
        """Create a heatmap showing content similarity between sections"""
        if not similarity_matrix or not section_titles:
            return None
        
        # Truncate long titles
        short_titles = [title[:20] + '...' if len(title) > 20 else title for title in section_titles]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=short_titles,
            y=short_titles,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Section Content Similarity Matrix",
            xaxis_title="Sections",
            yaxis_title="Sections",
            height=max(400, len(section_titles) * 25)
        )
        
        return fig
    
    def create_content_clusters(self, sections, clusters):
        """Create a visualization of content clusters"""
        if not sections or not clusters:
            return None
        
        df = pd.DataFrame({
            'section': [s['title'][:30] + '...' if len(s['title']) > 30 else s['title'] for s in sections],
            'word_count': [s['word_count'] for s in sections],
            'page': [s['page'] for s in sections],
            'cluster': clusters
        })
        
        fig = px.scatter(
            df, 
            x='page', 
            y='word_count',
            color='cluster',
            hover_name='section',
            title="Content Clusters by Page and Word Count",
            labels={'page': 'Page Number', 'word_count': 'Word Count'},
            color_continuous_scale='Set3'
        )
        
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='white')))
        
        return fig
    
    def create_readability_chart(self, readability_data):
        """Create a chart showing readability metrics"""
        if not readability_data:
            return None
        
        metrics = []
        values = []
        descriptions = []
        
        if 'flesch_reading_ease' in readability_data:
            metrics.append('Flesch Reading Ease')
            values.append(readability_data['flesch_reading_ease'])
            descriptions.append('Higher = Easier to read')
        
        if 'flesch_kincaid_grade' in readability_data:
            metrics.append('Flesch-Kincaid Grade')
            values.append(readability_data['flesch_kincaid_grade'])
            descriptions.append('Grade level required')
        
        if 'automated_readability_index' in readability_data:
            metrics.append('Automated Readability Index')
            values.append(readability_data['automated_readability_index'])
            descriptions.append('Grade level required')
        
        if not metrics:
            return None
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                text=[f"{v:.1f}" for v in values],
                textposition='auto',
                marker_color=self.colors[:len(metrics)],
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<br>%{customdata}<extra></extra>',
                customdata=descriptions
            )
        ])
        
        fig.update_layout(
            title="Document Readability Analysis",
            xaxis_title="Readability Metrics",
            yaxis_title="Score",
            height=400
        )
        
        return fig
    
    def create_statistics_overview(self, statistics):
        """Create an overview of document statistics"""
        if not statistics:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Word Distribution', 'Document Overview', 'Section Analysis', 'Reading Time'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # Word distribution (placeholder data)
        fig.add_trace(
            go.Bar(x=['Total Words', 'Total Sentences', 'Total Sections'], 
                   y=[statistics.get('total_words', 0), 
                      statistics.get('total_sentences', 0), 
                      statistics.get('total_sections', 0)],
                   marker_color=self.colors[:3]),
            row=1, col=1
        )
        
        # Document overview indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=statistics.get('total_words', 0),
                title={"text": "Total Words"},
                number={'font': {'size': 40}}
            ),
            row=1, col=2
        )
        
        # Section analysis pie chart
        if statistics.get('total_sections', 0) > 0:
            fig.add_trace(
                go.Pie(
                    labels=['Content', 'Structure'],
                    values=[statistics.get('total_words', 0), statistics.get('total_sections', 0) * 10],
                    hole=0.3
                ),
                row=2, col=1
            )
        
        # Reading time indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=statistics.get('reading_time_minutes', 0),
                title={"text": "Est. Reading Time (min)"},
                number={'font': {'size': 40}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Document Statistics Overview")
        
        return fig
    
    def create_relationship_network(self, relationships):
        """Create a network graph showing section relationships"""
        if not relationships or not relationships.get('related_sections'):
            return None
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges
        for section_data in relationships['related_sections']:
            section_title = section_data['section']
            G.add_node(section_title)
            
            for related in section_data['related']:
                related_title = related['title']
                similarity = related['similarity']
                
                if similarity > 0.4:  # Only show strong relationships
                    G.add_edge(section_title, related_title, weight=similarity)
        
        if len(G.nodes()) == 0:
            return None
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"Similarity: {weight:.3f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[:20] + '...' if len(node) > 20 else node)
            
            # Count connections
            adjacencies = list(G.neighbors(node))
            node_info.append(f"Section: {node}<br>Connections: {len(adjacencies)}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Section Relationship Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Sections with similar content are connected",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_wordcloud(self, text):
        """Create a word cloud from the document text"""
        if not text or len(text.strip()) == 0:
            return None
        
        try:
            # Create word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            # Convert to base64 for display
            img = io.BytesIO()
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            
            img.seek(0)
            img_b64 = base64.b64encode(img.read()).decode()
            
            return f"data:image/png;base64,{img_b64}"
        
        except Exception as e:
            print(f"Error creating word cloud: {e}")
            return None
