import os
import yaml
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_config(config_path=None):
    """
    Load configuration from a YAML file or create default config
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        # Default configuration
        config = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            },
            "rag": {
                "vector_store": "chroma",
                "embedding_model": "openai",
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
            "web_search": {
                "enabled": True,
                "max_results": 10,
            },
            "data_sources": {
                "fda": {
                    "api_enabled": True,
                    "api_key": os.getenv("FDA_API_KEY", ""),
                    "base_url": "https://api.fda.gov",
                    "endpoints": {
                        "drug": "drug/drugsfda.json",  # For approved drugs
                        "event": "drug/event.json",    # For adverse events
                        "label": "drug/label.json",    # For drug labels
                        "ndc": "drug/ndc.json"         # For National Drug Code Directory
                    },
                    "note": "For better results with FDA API, register for a key at https://api.fda.gov/signup"
                },
                "clinical_trials": {
                    "api_enabled": True,
                    "base_url": "https://clinicaltrials.gov/api",
                },
                "pubmed": {
                    "api_enabled": True,
                    "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
                    "api_key": os.getenv("PUBMED_API_KEY", ""),
                },
                "custom_sources": []
            },
            "cache": {
                "enabled": True,
                "ttl": 86400,  # 24 hours
            },
            "logging": {
                "level": "INFO",
                "file": "drug_approval_agent.log",
            }
        }
    
    return config

def save_config(config, config_path):
    """
    Save configuration to a YAML file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return True

# Helper function to create .env file template
def create_env_template(output_path=".env.template"):
    """
    Create a template .env file with required variables
    """
    env_vars = [
        "OPENAI_API_KEY=your_openai_api_key_here",
        "OPENAI_MODEL=gpt-4o",
        "FDA_API_KEY=your_fda_api_key_here",
        "PUBMED_API_KEY=your_pubmed_api_key_here",
        "CHROMADB_PERSIST_DIR=./data/chroma",
        "CACHE_DIR=./data/cache",
        "LOG_LEVEL=INFO",
    ]
    
    with open(output_path, 'w') as file:
        file.write("\n".join(env_vars))
    
    return True 