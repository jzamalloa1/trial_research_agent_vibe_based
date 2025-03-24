# Drug Approval Intelligence

An agentic AI tool that provides insights into drug approvals and pharmaceutical development pipelines using LangChain, OpenAI, and modern web technologies.

## Overview

This project provides a comprehensive solution for accessing and analyzing information about drug approvals, clinical trials, and pharmaceutical developments. It includes a modern dark-themed user interface where users can ask questions and get visualized insights backed by data from multiple sources.

![Drug Approval Intelligence Screenshot - Placeholder](https://via.placeholder.com/800x450?text=Drug+Approval+Intelligence)

## Key Features

- **AI-Powered Research**: Ask natural language questions about drugs, approvals, clinical trials
- **Dynamic Visualizations**: Interactive charts and graphs showing drug pipeline statistics
- **Comprehensive Data**: Information about manufacturers, molecules, pipeline stages, and news mentions
- **Modern UI**: Dark mode React components with responsive design

## Technologies Used

- **LangChain**: For building the agentic AI framework
- **OpenAI**: For language model integration (optional: Responses API support)
- **RAG (Retrieval Augmented Generation)**: For enhancing responses with domain-specific knowledge
- **Streamlit**: For the web interface
- **Plotly**: For interactive data visualizations
- **Python Backend**: For data processing and API integration

## Getting Started

See the [Drug Approval Intelligence README](./drug_approval_agent/README.md) for detailed setup and usage instructions.

## Project Structure

```
drug_approval_agent/
├── api/                  # API interfaces and data sources
├── components/           # UI components (when migrating to React)
├── data/                 # Data storage
├── models/               # ML model definitions
├── utils/                # Utility functions
├── app.py                # Main Streamlit application
├── environment.yml       # Conda environment definition
└── README.md             # Detailed documentation
```

## License

MIT 