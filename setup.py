#!/usr/bin/env python3
"""Setup script for Concept Bottleneck Models project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Concept Bottleneck Models project...")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e '.[dev]'", "Installing development dependencies"):
        sys.exit(1)
    
    # Create necessary directories
    directories = ["logs", "outputs", "assets", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run tests
    if not run_command("pytest tests/ -v", "Running unit tests"):
        print("⚠️  Some tests failed, but continuing with setup...")
    
    # Run linting
    if not run_command("ruff check src/ tests/ scripts/", "Running code linting"):
        print("⚠️  Some linting issues found, but continuing with setup...")
    
    print("=" * 60)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/app.py")
    print("2. Train a model: python scripts/train.py")
    print("3. Explore the notebook: jupyter notebook notebooks/quickstart.ipynb")
    print("\n⚠️  Remember: This tool is for research and educational purposes only!")
    print("   XAI outputs may be unstable or misleading. Not a substitute for human judgment.")


if __name__ == "__main__":
    main()
