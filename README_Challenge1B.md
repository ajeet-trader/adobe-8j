# Adobe India Hackathon 2025 - Challenge 1B Solution

## Overview

This is a complete end-to-end solution for Challenge 1B: Multi-Collection PDF Analysis. The solution implements sophisticated persona-based content analysis across multiple document collections.

## Features

- **Persona-Based Analysis**: Tailored content extraction for Travel Planner, HR Professional, and Food Contractor personas
- **Advanced Relevance Scoring**: Multi-factor scoring algorithm considering persona keywords, task requirements, and content quality
- **Cross-Document Analysis**: Ensures balanced representation across all input documents
- **Content Refinement**: Persona-specific text processing and refinement
- **Comprehensive Validation**: Built-in testing and validation framework

## Quick Start

### Option 1: Direct Execution
\`\`\`bash
# Install dependencies
pip install -r requirements_1b.txt

# Run the solution
python challenge1b_solution.py

# Validate results
python test_challenge1b.py
\`\`\`

### Option 2: Using the Runner Script
\`\`\`bash
chmod +x run_challenge1b.sh
./run_challenge1b.sh
\`\`\`

### Option 3: Docker Deployment
\`\`\`bash
# Build the container
docker build -f Dockerfile_1b -t challenge1b-processor .

# Run with volume mounts
docker run --rm \
  -v $PWD/Collection\ 1:/app/Collection\ 1 \
  -v $PWD/Collection\ 2:/app/Collection\ 2 \
  -v $PWD/Collection\ 3:/app/Collection\ 3 \
  challenge1b-processor
\`\`\`

## Directory Structure

\`\`\`
Challenge_1b/
├── Collection 1/
│   ├── PDFs/                    # Travel guide PDFs
│   ├── challenge1b_input.json   # Input configuration
│   └── challenge1b_output.json  # Generated output
├── Collection 2/
│   ├── PDFs/                    # Acrobat tutorial PDFs
│   ├── challenge1b_input.json   # Input configuration
│   └── challenge1b_output.json  # Generated output
├── Collection 3/
│   ├── PDFs/                    # Recipe guide PDFs
│   ├── challenge1b_input.json   # Input configuration
│   └── challenge1b_output.json  # Generated output
├── challenge1b_solution.py      # Main solution
├── test_challenge1b.py          # Validation script
├── requirements_1b.txt          # Dependencies
└── README_Challenge1B.md        # This file
\`\`\`

## Algorithm Details

### Persona-Based Keyword Mapping

The solution uses sophisticated keyword mapping for each persona:

- **Travel Planner**: Prioritizes itinerary, accommodation, activities, and practical travel information
- **HR Professional**: Focuses on forms, compliance, workflows, and step-by-step processes
- **Food Contractor**: Emphasizes recipes, ingredients, dietary restrictions, and catering requirements

### Relevance Scoring Algorithm

The relevance scoring combines multiple factors:

1. **Persona Keywords** (70% weight): High/medium/low priority keyword matching
2. **Task-Specific Terms** (30% weight): Keywords extracted from the specific task description
3. **Content Quality Boost**: Length-based scoring for substantial content

### Content Refinement

Each persona receives tailored content refinement:

- **Travel Planner**: Extracts actionable travel information, costs, and recommendations
- **HR Professional**: Focuses on step-by-step instructions and form-related processes
- **Food Contractor**: Prioritizes recipes, ingredients, and cooking instructions

## Output Quality Metrics

The solution ensures high-quality output through:

- **Balanced Document Representation**: Content from all input documents
- **Importance Ranking**: Genuine relevance-based ranking
- **Content Filtering**: Meaningful text extraction with noise reduction
- **Schema Compliance**: Exact adherence to required JSON structure

## Validation Framework

The included validation script checks:

- ✅ JSON schema compliance
- ✅ Content quality metrics
- ✅ Ranking validity
- ✅ Document coverage
- ✅ Persona relevance

## Expected Results

### Collection 1 (Travel Planning)
- Extracts coastal activities, culinary experiences, accommodation options
- Prioritizes group-friendly activities and budget information
- Focuses on 4-day itinerary planning for South of France

### Collection 2 (HR Professional)
- Emphasizes fillable form creation and management
- Extracts step-by-step compliance processes
- Prioritizes onboarding workflow automation

### Collection 3 (Food Contractor)
- Focuses on vegetarian and gluten-free recipes
- Extracts buffet-suitable dishes and serving information
- Prioritizes corporate catering requirements

## Troubleshooting

### Common Issues

1. **Missing PDFs**: Place PDF files in respective `PDFs/` directories
2. **Low Relevance Scores**: Adjust keyword mappings in `persona_keywords`
3. **Schema Validation Errors**: Check JSON structure against expected format
4. **Performance Issues**: Consider reducing text processing for large documents

### Performance Optimization

- The solution processes up to 50 top sections and 100 subsections per collection
- Text blocks under 10 characters are filtered out
- Relevance threshold of 0.2 for inclusion

## Contributing

To enhance the solution:

1. Add new persona types in `persona_keywords`
2. Improve relevance scoring algorithms
3. Enhance content refinement strategies
4. Add support for additional document formats

## License

This solution is developed for the Adobe India Hackathon 2025. Please refer to the hackathon terms and conditions for usage rights.
