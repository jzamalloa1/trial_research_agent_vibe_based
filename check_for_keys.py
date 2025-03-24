#!/usr/bin/env python
"""
Scan the codebase for potential API keys or credentials before making commits
Run this script before pushing to a repository to ensure no sensitive data is leaked
"""
import os
import re
import sys
from pathlib import Path

# Patterns that might indicate API keys or credentials
API_KEY_PATTERNS = [
    r'(api[_-]?key|apikey|key|token|secret|password|credential)["\']?\s*(?::|=>|=)\s*["\']?([a-zA-Z0-9._\-]{16,})["\']?',
    r'sk-[a-zA-Z0-9]{30,}',  # OpenAI API key pattern
    r'github_pat_[a-zA-Z0-9]{20,}',  # GitHub token pattern
    r'Bearer\s+[a-zA-Z0-9._~+/=-]{30,}',  # Bearer token pattern
]

# Additional patterns for environment files
ENV_KEY_PATTERNS = [
    r'([A-Z0-9_]+)=([a-zA-Z0-9._\-]{16,})',  # ENV_VAR=longstring
    r'([A-Z0-9_]+)="([a-zA-Z0-9._\-]{16,})"',  # ENV_VAR="longstring"
    r"([A-Z0-9_]+)='([a-zA-Z0-9._\-]{16,})'",  # ENV_VAR='longstring'
]

# List of files and directories to exclude
EXCLUDED_PATHS = [
    '.git', 'node_modules', 'venv', '.venv', 'env', 
    '__pycache__', '.idea', '.vscode', '.DS_Store',
    'check_for_keys.py', 'pre-push.py', '*.log', 'drug_approval_agent.log'
]

# Keywords that indicate a placeholder rather than a real key
PLACEHOLDER_KEYWORDS = [
    'your_api_key', 'your-api-key', 'api_key_here', 'placeholder',
    'your_', 'example', 'sample', 'test', 'demo', 'dummy', 'fake'
]

def is_excluded(path):
    """Check if path should be excluded from scanning"""
    path_str = str(path)
    return any(excluded in path_str for excluded in EXCLUDED_PATHS)

def is_binary(file_path):
    """Check if file is binary"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
            return False
    except UnicodeDecodeError:
        return True

def is_placeholder(text):
    """Check if a string is likely a placeholder rather than a real key"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in PLACEHOLDER_KEYWORDS)

def check_file(file_path):
    """Check a single file for API keys"""
    if is_binary(file_path):
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return []

    findings = []
    
    # Determine if this is an environment file
    is_env_file = file_path.endswith(('.env', '.env.local', '.env.development', '.env.production'))
    patterns_to_check = API_KEY_PATTERNS
    
    # For .env files, also check env-specific patterns
    if is_env_file and not file_path.endswith('.env.template'):
        patterns_to_check = patterns_to_check + ENV_KEY_PATTERNS
    
    # Check each pattern
    for pattern in patterns_to_check:
        matches = re.finditer(pattern, content)
        for match in matches:
            matched_text = match.group(0)
            
            # Skip if it's clearly a placeholder
            if is_placeholder(matched_text):
                continue
                
            # Get the line number and content
            line_number = content[:match.start()].count('\n') + 1
            line_content = content.splitlines()[line_number-1]
            
            findings.append({
                'file': file_path,
                'line': line_number,
                'content': line_content.strip(),
                'key': matched_text
            })
            
    return findings

def scan_directory(directory='.'):
    """Recursively scan a directory for files with potential API keys"""
    findings = []
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d))]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip excluded files and directories
            if is_excluded(file_path):
                continue
            
            # Skip common binary and non-text files
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.pyc')):
                continue
                
            # Check for .env files (except .env.template)
            if file.endswith('.env') and not file.endswith('.env.template'):
                print(f"Found .env file: {file_path}")
                
            # Check the file content
            file_findings = check_file(file_path)
            if file_findings:
                findings.extend(file_findings)
    
    return findings

def main():
    """Main function"""
    directory = '.'
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    print(f"Scanning directory: {directory}")
    findings = scan_directory(directory)
    
    if findings:
        print("\n⚠️  POTENTIAL SECURITY RISKS FOUND ⚠️\n")
        for finding in findings:
            print(f"File: {finding['file']}")
            if finding['line'] > 0:
                print(f"Line: {finding['line']}")
            print(f"Content: {finding['content']}")
            print("-" * 80)
        
        print(f"\nFound {len(findings)} potential security risks. Please review before committing.")
        return 1
    else:
        print("✅ No API keys or credentials found. You're good to go!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 