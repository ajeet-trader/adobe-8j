# Adobe India Hackathon 2025 - Challenge 1B Streamlit Solution

## 🎯 Overview

This is a complete Streamlit-based solution for Challenge 1B: Multi-Collection PDF Analysis. The application provides an intuitive web interface for persona-based content analysis across multiple document collections.

## 🚀 Quick Start

### Step 1: Install Dependencies
\`\`\`bash
pip install -r requirements_challenge1b.txt
\`\`\`

### Step 2: Run the Application
\`\`\`bash
streamlit run challenge1b_app.py
\`\`\`

### Step 3: Access the Web Interface
Open your browser and navigate to `http://localhost:8501`

## 📚 Supported Collections

### 1. Travel Planning Collection
- **Persona:** Travel Planner
- **Task:** Plan a 4-day trip for 10 college friends to South of France
- **Expected Documents:** Cities.pdf, Cuisine.pdf, History.pdf, Restaurants & Hotels.pdf, Things to Do.pdf, Tips & Tricks.pdf, Traditions & Culture.pdf

### 2. HR Forms Management Collection
- **Persona:** HR Professional
- **Task:** Create and manage fillable forms for onboarding and compliance
- **Expected Documents:** Creating Fillable Forms.pdf, Form Field Properties.pdf, Digital Signatures.pdf, Document Security.pdf, Workflow Automation.pdf

### 3. Corporate Catering Collection
- **Persona:** Food Contractor
- **Task:** Prepare vegetarian buffet-style dinner menu for corporate gathering with gluten-free items
- **Expected Documents:** Breakfast Recipes.pdf, Lunch Mains.pdf, Dinner Mains.pdf, Side Dishes.pdf, Gluten-Free Options.pdf, Vegetarian Specialties.pdf

## 🎨 Application Features

### 📊 Interactive Dashboard
- Real-time analysis progress tracking
- Comprehensive overview metrics
- Document processing statistics

### 🏆 Advanced Section Ranking
- Persona-specific relevance scoring
- Task-oriented content prioritization
- Balanced document representation

### 📈 Rich Visualizations
- Section distribution charts
- Importance ranking histograms
- Page-wise content analysis
- Relevance score distributions

### 💾 Multiple Export Formats
- Complete analysis (JSON)
- Structured sections (CSV)
- Summary reports (Markdown)

## 🔧 Technical Architecture

### Persona-Based Analysis Engine
The solution implements sophisticated persona-based content analysis with:

- **Multi-tier Keyword Mapping**: High/medium/low priority keywords for each persona
- **Task-Specific Scoring**: Dynamic keyword extraction from task descriptions
- **Content Quality Metrics**: Length-based scoring and meaningful text filtering
- **Cross-Document Balancing**: Ensures representation from all input documents

### Advanced Text Processing
- **Intelligent Section Detection**: Font-size and structure-based title identification
- **Content Refinement**: Persona-specific text processing and filtering
- **Relevance Calculation**: Multi-factor scoring combining persona and task relevance

## 📖 How to Use

### Step 1: Collection Selection
1. Choose from three predefined collections
2. Review the persona and task description
3. Note the expected document types

### Step 2: Document Upload
1. Upload multiple PDF files for the selected collection
2. Verify all expected documents are included
3. Review the uploaded file list

### Step 3: Analysis Execution
1. Click "Analyze Collection" to start processing
2. Monitor real-time progress updates
3. Wait for analysis completion

### Step 4: Results Exploration
Navigate through five comprehensive tabs:

#### 📈 Overview Tab
- Key metrics and statistics
- Collection details and processing information
- Task description and input documents

#### 🏆 Top Sections Tab
- Ranked list of most relevant sections
- Document and rank filtering options
- Detailed section information

#### 📝 Content Analysis Tab
- Document-wise content breakdown
- Relevance-scored content items
- Refined text extractions

#### 📊 Visualizations Tab
- Interactive charts and graphs
- Distribution analysis
- Relevance score visualization

#### 💾 Export Tab
- Multiple export format options
- Data preview capabilities
- Summary report generation

## 🎯 Quality Assurance

### Content Validation
- **Schema Compliance**: Ensures output matches expected JSON structure
- **Relevance Verification**: Validates persona-specific content extraction
- **Ranking Accuracy**: Confirms importance ranking reflects actual relevance

### Performance Optimization
- **Efficient Processing**: Optimized PDF text extraction and analysis
- **Memory Management**: Handles large document collections efficiently
- **Progress Tracking**: Real-time feedback during processing

## 🔍 Algorithm Details

### Relevance Scoring Formula
\`\`\`
Final Score = (Persona Score × 0.7) + (Task Score × 0.3) + Length Boost
\`\`\`

Where:
- **Persona Score**: Weighted keyword matching based on priority levels
- **Task Score**: Task-specific keyword relevance
- **Length Boost**: Up to 20% boost for substantial content

### Content Refinement Strategies

#### Travel Planner
- Prioritizes actionable travel information
- Extracts cost and timing details
- Focuses on group-friendly activities

#### HR Professional
- Emphasizes step-by-step instructions
- Extracts form-related processes
- Prioritizes compliance workflows

#### Food Contractor
- Focuses on recipes and ingredients
- Extracts cooking instructions
- Prioritizes dietary restrictions

## 🚀 Advanced Features

### Real-Time Processing
- Live progress updates during analysis
- Dynamic status messages
- Responsive user interface

### Interactive Filtering
- Document-based filtering
- Rank range selection
- Dynamic data visualization

### Comprehensive Export
- JSON for complete analysis
- CSV for structured data
- Markdown for readable reports

## 🛠️ Troubleshooting

### Common Issues

1. **PDF Upload Errors**
   - Ensure PDFs are not password-protected
   - Check file size limits
   - Verify PDF format compatibility

2. **Low Relevance Scores**
   - Review persona keyword mappings
   - Adjust relevance thresholds
   - Verify task description accuracy

3. **Processing Delays**
   - Large documents may take longer
   - Monitor progress indicators
   - Consider document size optimization

### Performance Tips
- Upload documents in smaller batches for faster processing
- Use high-quality, text-based PDFs for better extraction
- Ensure stable internet connection for Streamlit interface

## 📊 Expected Results

### Travel Planning Collection
- Extracts itinerary suggestions and activity recommendations
- Prioritizes group-friendly attractions and budget information
- Focuses on South of France specific content

### HR Forms Collection
- Emphasizes fillable form creation processes
- Extracts compliance and onboarding workflows
- Prioritizes step-by-step Adobe Acrobat instructions

### Corporate Catering Collection
- Focuses on vegetarian and gluten-free recipes
- Extracts buffet-suitable dishes and serving information
- Prioritizes corporate event catering requirements

## 🎉 Success Metrics

A successful analysis should demonstrate:
- ✅ High relevance scores for persona-specific content
- ✅ Balanced representation across all input documents
- ✅ Meaningful section titles and content extractions
- ✅ Accurate importance ranking based on persona needs
- ✅ Clean, actionable content in subsection analysis

---

**Built for Adobe India Hackathon 2025 - Challenge 1B**
*Streamlit-powered persona-based PDF analysis solution*
