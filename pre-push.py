#!/usr/bin/env python
"""
Pre-push check script to ensure no API keys are accidentally committed
Usage: Copy this file to .git/hooks/pre-push and make it executable (chmod +x .git/hooks/pre-push)
"""
import os
import re
import sys
import subprocess

def check_for_api_keys():
    """Check for API keys in staged files"""
    # Patterns to look for
    api_key_patterns = [
        r'[a-zA-Z0-9]{32,}',  # Generic API key pattern
        r'sk-[a-zA-Z0-9]{30,}',  # OpenAI API key pattern
        r'github_pat_[a-zA-Z0-9]{20,}',  # GitHub token pattern
        r'Bearer\s+[a-zA-Z0-9._~+/=-]{30,}',  # Bearer token pattern
    ]
    
    # Get staged files
    try:
        staged_files = subprocess.check_output(
            ['git', 'diff', '--name-only', '--cached'],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        ).splitlines()
    except subprocess.CalledProcessError:
        print("Error getting staged files. Are you in a git repository?")
        return False
    
    found_keys = False
    
    # Check each staged file
    for file_path in staged_files:
        # Skip binary files, .env.template is allowed
        if not os.path.isfile(file_path) or file_path.endswith('.env.template'):
            continue
        
        # Skip files that should contain API keys (like this checker script)
        if file_path == os.path.basename(__file__):
            continue
            
        # Check for .env files that shouldn't be committed
        if file_path.endswith('.env') and not file_path.endswith('.env.template'):
            print(f"WARNING: You're trying to commit a .env file: {file_path}")
            found_keys = True
            continue
            
        # Read file contents
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            print(f"Could not read file: {file_path} (might be binary)")
            continue
            
        # Check for API key patterns
        for pattern in api_key_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                matched_text = match.group(0)
                
                # Skip if it's clearly a placeholder
                if 'your_api_key' in matched_text.lower() or \
                   'your-api-key' in matched_text.lower() or \
                   'api_key_here' in matched_text.lower() or \
                   'placeholder' in matched_text.lower():
                    continue
                    
                # Skip if it's in an example block or code comment
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if matched_text in line:
                        is_example = False
                        # Check if in a comment
                        if '#' in line and line.index('#') < line.index(matched_text):
                            is_example = True
                        if '//' in line and line.index('//') < line.index(matched_text):
                            is_example = True
                        if is_example:
                            continue
                        
                        # Alert about potential API key
                        print(f"WARNING: Possible API key found in {file_path}:{i+1}")
                        print(f"  {line.strip()}")
                        found_keys = True
                        break
    
    return not found_keys

if __name__ == "__main__":
    print("Running pre-push hook to check for API keys...")
    if not check_for_api_keys():
        print("Potential API keys found in staged files. Please review the warnings above.")
        print("To force push anyway, use 'git push --no-verify'")
        sys.exit(1)
    else:
        print("No API keys found. Proceeding with push.")
        sys.exit(0) 