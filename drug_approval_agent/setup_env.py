#!/usr/bin/env python
"""
Setup environment script - copies the .env.template to .env
"""
import os
import shutil
import sys

def setup_env():
    """Copy template env file to .env if it doesn't exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, '.env.template')
    env_path = os.path.join(script_dir, '.env')
    
    if not os.path.exists(template_path):
        print(f"Error: Template file {template_path} not found.")
        return False
    
    if os.path.exists(env_path):
        overwrite = input(f"File {env_path} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Setup cancelled.")
            return False
    
    try:
        shutil.copy2(template_path, env_path)
        print(f"Created {env_path} from template.")
        print("Please edit this file to add your actual API keys.")
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

if __name__ == "__main__":
    if setup_env():
        print("Environment setup complete!")
    else:
        print("Environment setup failed.")
        sys.exit(1) 