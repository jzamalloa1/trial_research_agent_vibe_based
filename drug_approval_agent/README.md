# Drug Approval Intelligence

An AI-powered tool for exploring drug approvals, clinical trials, and pharmaceutical development pipelines with a modern dark-themed UI.

## Features

- **AI Assistant**: Ask natural language questions about drugs, approvals, and pharmaceutical companies
- **Rich Data Visualization**: Dynamic charts, timelines, and data tables
- **Multiple Data Sources**: Integrated FDA approvals, clinical trials, PubMed publications, and news
- **Modern UI**: Dark mode, responsive design, and intuitive interface

## ⚠️ Important API Notice

**Warning**: The Clinical Trials API integration requires additional configuration to function correctly. The current implementation may encounter issues with query formatting and endpoint compatibility. Further development is needed to:

1. Properly parse and format natural language queries for the Clinical Trials API
2. Handle pagination and response formats correctly
3. Implement more robust error handling for API-specific response codes

If you encounter 404 errors or no results when querying clinical trials data, this is a known issue that needs to be addressed.

## Architecture

This tool uses a combination of:

- **LangChain**: Framework for building agentic AI applications
- **OpenAI**: For natural language processing and question answering
- **Streamlit**: For the web interface and visualizations
- **Plotly**: For interactive data visualizations
- **Python Backend**: For data retrieval, processing, and API integration

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (for environment management)
- OpenAI API key

### Installation

1. Clone this repository

2. Create and activate the conda environment:
   ```bash
   cd drug_approval_agent
   conda env create -f environment.yml
   conda activate cursor_python
   ```

3. Set up environment variables:
   ```bash
   cp .env.template .env
   ```
   Then edit the `.env` file to add your API keys.

### API Keys and Security

This application requires several API keys to function properly:

- **OpenAI API Key**: Required for the AI assistant functionality
- **FDA API Key**: For accessing FDA drug databases (optional but recommended)
- **PubMed API Key**: For fetching research papers (optional)

API keys are stored in the `.env` file, which is excluded from version control in the `.gitignore` file. 
Never commit your `.env` file containing real API keys to the repository.

When sharing or distributing this code:
- Only use the `.env.template` file which contains placeholders
- Ensure the `.gitignore` file is properly set up to exclude the `.env` file
- Verify no API keys are hardcoded in any source files before pushing

#### Pre-Push Safety Check

For additional security, you can set up a pre-push git hook to automatically check for API keys:

```bash
# Copy the pre-push script to your git hooks directory
cp ../pre-push.py .git/hooks/pre-push
# Make it executable
chmod +x .git/hooks/pre-push
```

This hook will scan staged files before pushing to ensure no API keys or credentials are accidentally committed.

#### Security Scan Script

You can also run the provided script to scan the entire codebase for potential API keys:

```bash
python ../check_for_keys.py
```

Run this before your first commit or periodically to ensure no credentials have been accidentally added to your codebase.

### Running the Application

To start the application, run:

```bash
cd drug_approval_agent
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

There are two main ways to interact with the system:

1. **Search Interface**: Use the sidebar to search for drugs by name, manufacturer, therapeutic area, or approval status.

2. **AI Assistant**: Ask natural language questions about drug approvals, pharmaceutical companies, and clinical trials.

Example questions:
- "What are the latest FDA approved drugs for cancer?"
- "What's in Pfizer's drug pipeline for 2023?"
- "Which drugs have been approved for Alzheimer's in the last 2 years?"
- "What is the status of CAR-T therapies in clinical trials?"
- "Which companies are leading in rare disease drug development?"

## Data Sources

- FDA Drug Approval Database
- ClinicalTrials.gov *(⚠️ API integration needs additional development - see warning above)*
- PubMed
- News and web sources

When the Clinical Trials API fails, the system falls back to web search results, which may not be as comprehensive or structured as direct API data.

## Extending the System

The modular architecture makes it easy to extend:

1. Add new data sources in the `api` directory
2. Create new visualizations in the Streamlit app
3. Enhance the AI agent with additional tools or capabilities

## Known Issues

1. **Clinical Trials API Integration**: The current implementation has issues properly formatting natural language queries for the ClinicalTrials.gov API. In many cases, 404 errors are returned because the API doesn't accept direct natural language questions as search parameters.

2. **API Error Handling**: Error handling for Clinical Trials API still needs improvement. Currently, the system falls back to web search when API queries fail, which provides less structured data.

3. **Query Generation**: The extraction of key terms from natural language queries needs refinement for better matching with the Clinical Trials API's expected parameters.

## License

MIT 