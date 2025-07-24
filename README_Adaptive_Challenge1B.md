# 🧠 Adaptive Smart PDF Tool - Challenge 1B

## 🎯 Overview

This is a **truly adaptive** solution for Adobe India Hackathon 2025 Challenge 1B that can handle **ANY persona and task**, not just predefined ones. The tool dynamically analyzes your persona and task description to adapt its algorithms, keyword matching, and content extraction strategies in real-time.

## 🧠 Key Adaptive Features

### Dynamic Persona Analysis
- **Automatic Domain Detection**: Identifies whether you're in technology, healthcare, finance, legal, etc.
- **Keyword Extraction**: Dynamically extracts relevant terms from your persona description
- **Role Understanding**: Analyzes your professional role and responsibilities

### Intelligent Task Processing
- **Action Verb Identification**: Finds key actions you need to perform
- **Context Clue Extraction**: Identifies numbers, timeframes, and scale indicators
- **Priority Concept Mapping**: Determines what concepts are most important for your task

### Adaptive Relevance Scoring
The tool uses a sophisticated multi-factor scoring system:
- **Persona Keywords** (30%): Terms specific to your role
- **Task Keywords** (25%): Terms from your task description  
- **Action Verbs** (20%): Action-oriented content matching your needs
- **Priority Concepts** (15%): Important concepts for your domain
- **Context Clues** (10%): Contextual information like timeframes, scale

## 🚀 How to Use

### Step 1: Define Your Persona
Enter **any** professional role or persona:
- Data Scientist analyzing research papers
- Legal Advisor reviewing contracts
- Marketing Manager extracting campaign insights
- Financial Analyst studying investment reports
- Product Manager gathering requirements
- Consultant finding best practices

### Step 2: Describe Your Task
Be specific about what you want to accomplish:
- "Extract key findings for quarterly board presentation"
- "Identify compliance requirements for new product launch"
- "Find competitive analysis insights for strategy meeting"
- "Summarize technical requirements for development team"

### Step 3: Upload Documents
Upload any PDF documents relevant to your analysis.

### Step 4: Adaptive Analysis
The tool will:
1. **Analyze** your persona and task
2. **Adapt** its algorithms dynamically
3. **Extract** the most relevant content
4. **Rank** by adaptive importance
5. **Present** tailored insights

## 🎨 User Interface Features

### Real-Time Adaptation Display
- Shows detected domains and focus areas
- Displays key terms identified from your input
- Previews action focus before analysis

### Enhanced Results Presentation
- **Adaptive Overview**: Shows how the tool adapted to your needs
- **Smart Ranking**: Sections ranked by true relevance to your persona/task
- **Relevance Factors**: Shows why each section was considered relevant
- **Content Categories**: High/medium/low relevance classification

### Interactive Analysis
- Filter by relevance factors
- Explore adaptation effectiveness
- View document contribution analysis
- Get personalized recommendations

## 🔧 Technical Architecture

### Adaptive Algorithm Engine
```python
# Dynamic persona analysis
persona_analysis = analyze_persona_and_task(persona, task)

# Adaptive relevance calculation
relevance_score = calculate_adaptive_relevance(text, persona_analysis)

# Domain-specific content refinement
refined_content = refine_text_adaptive(text, persona_analysis)
