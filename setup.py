#!/usr/bin/env python3

import os
import stat

# --- Configuration ---
PROJECT_NAME = "scholarpulse"

DIRECTORIES = [
    "core",
    "ui",
    "temp_data",
]

EMPTY_FILES = [
    os.path.join("core", "__init__.py"),
    os.path.join("core", "agents.py"),
    os.path.join("core", "llm_setup.py"),
    os.path.join("core", "parsing.py"),
    os.path.join("core", "retrieval.py"),
    os.path.join("core", "utils.py"),
    os.path.join("ui", "__init__.py"),
    os.path.join("ui", "components.py"),
    os.path.join("ui", "views.py"),
    os.path.join("temp_data", ".gitkeep"), # Keep dir in git
]

# --- File Contents ---

GITIGNORE_CONTENT = """\
# Environment variables
.env
*.env.*
!.env.example

# Temporary data/uploads
temp_data/

# Python cache & artifacts
__pycache__/
*.pyc
*.pyo
*.pyd
build/
dist/
*.egg-info/
*.spec

# Virtual environment
venv/
env/
.venv/

# IDE files
.idea/
.vscode/

# OS generated files
.DS_Store
Thumbs.db
"""

REQUIREMENTS_CONTENT = """\
# Core Frameworks
streamlit
langchain
langchain-groq

# LLM & Langchain Helpers
python-dotenv       # For loading .env files
langchainhub        # For pulling prompts/agents if needed

# Data Handling & Parsing
pypdf               # PDF parsing (or choose pymupdf, add to packages.txt if needed)
requests            # For fetching URLs/DOIs
beautifulsoup4      # For parsing HTML from URLs

# Vector Store & Embeddings (Choose one VDB or adapt)
chromadb            # Example local vector database
# faiss-cpu         # Alternative vector database (or faiss-gpu if you have CUDA)
sentence-transformers # Often used for generating embeddings locally

# Optional: Add others as needed
# pandas
# numpy
"""

PACKAGES_CONTENT = """\
# System dependencies needed for some Python packages.
# Add libraries required by packages like pymupdf or others if needed.
# Example for Debian/Ubuntu based systems on Hugging Face Spaces:
# poppler-utils
# build-essential
"""

ENV_EXAMPLE_CONTENT = """\
# Groq API Key (Get from https://console.groq.com/keys)
GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE

# Optional: Other API Keys if needed (e.g., Semantic Scholar, GitHub)
# GITHUB_TOKEN=YOUR_GITHUB_TOKEN_HERE
"""

APP_PY_CONTENT = """\
import streamlit as st
from dotenv import load_dotenv
import os
import sys

# Add core directory to sys.path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))


# Load environment variables (especially API keys) BEFORE importing core modules
# that might use them during initialization
load_dotenv()

# --- Import UI and Core functions (Example - adjust imports as you build) ---
# Uncomment these as you build the corresponding files
# from ui.components import display_sidebar, display_chat # Example
# from ui.views import render_summary_view, render_qa_view # Example
# from core.parsing import process_input # Example
# from core.retrieval import get_retriever # Example
# from core.agents import get_qa_agent # Example

# --- Page Configuration (Set Title, Icon, Layout) ---
st.set_page_config(
    page_title="ScholarPulse",
    page_icon="💡",
    layout="wide"
)

st.title("🎓 ScholarPulse: AI Research Assistant")
st.caption("Upload a paper (PDF) or enter a URL (arXiv, DOI) to get started.")

# --- Main Application Logic ---

# 1. Sidebar for Input & Options
with st.sidebar:
    st.header("Input Paper")
    # Add file uploader, text input for URL/DOI
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    source_input = st.text_input("Or Enter arXiv URL, DOI, or Publisher Link")

    st.header("Your Background")
    # Add selectbox for user background
    user_background = st.selectbox(
        "Select your background level:",
        ("Bachelor's Student", "Master's Student", "PhD Student/Researcher", "Industry Professional", "Curious Learner"),
        key="user_background" # Add a key for session state if needed later
    )

    process_button = st.button("Analyze Paper")

# Initialize session state for potentially storing results or intermediate data
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
# Add more session state variables as needed: e.g., paper_data, qa_agent


# 2. Main Area for Display
if process_button and (uploaded_file or source_input):
    with st.spinner("Processing paper... Please wait."):
        try:
            st.session_state.analysis_complete = False # Reset on new processing
            st.session_state.error_message = None
            # --- Placeholder: Add logic to call core functions ---
            # input_source = uploaded_file if uploaded_file else source_input
            # paper_data = process_input(input_source) # Your core function
            # if paper_data:
            #     st.session_state.paper_data = paper_data
            #     # retriever = get_retriever(paper_data) # Your core function
            #     # qa_agent = get_qa_agent(retriever, st.session_state.user_background) # Pass background
            #     # st.session_state.qa_agent = qa_agent
            st.session_state.analysis_complete = True # Set true on success
            st.success("Paper processed successfully! (Placeholder)") # Replace with actual success
            # else:
            #     st.session_state.error_message = "Failed to process the paper. Check input or logs."

        except Exception as e:
            st.session_state.error_message = f"An error occurred: {e}"
            st.error(st.session_state.error_message)
            print(f"Error during processing: {e}") # Log to console for debugging

elif process_button:
    st.warning("Please upload a PDF or enter a valid source URL/DOI.")


# 3. Display results if analysis is complete
if st.session_state.analysis_complete:
    st.markdown("--- Analysis Results ---")
    # --- Placeholder: Render different views using functions from ui/views.py ---
    # if 'paper_data' in st.session_state:
        # render_summary_view(st.session_state.paper_data) # Example view function call
    # if 'qa_agent' in st.session_state:
        # render_qa_view(st.session_state.qa_agent) # Example view function call
    st.write("Display summary, Q&A interface, code generation options here.") # Placeholder
elif st.session_state.error_message:
    # Error already displayed above, or you can ensure it's displayed here
    pass # Optionally add more context here if needed
else:
    st.info("Upload a paper or enter a source above and click 'Analyze Paper'.")


# --- Example Footer ---
st.markdown("---")
st.caption("Developed with LangChain, Groq, and Streamlit.")

"""

# --- Main Script Logic ---
def create_file(path, content=""):
    """Creates a file with the given content."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Created file: {path}")
    except IOError as e:
        print(f"Error creating file {path}: {e}")

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"  Created directory: {path}")
    except OSError as e:
        print(f"Error creating directory {path}: {e}")

def main():
    """Main function to set up the project structure."""
    print(f"Setting up project '{PROJECT_NAME}'...")

    if os.path.exists(PROJECT_NAME):
        print(f"Error: Directory '{PROJECT_NAME}' already exists. Please remove it or choose a different name.")
        return

    # Create root project directory
    create_directory(PROJECT_NAME)
    os.chdir(PROJECT_NAME) # Change into the project directory

    # Create subdirectories
    print("\nCreating directories...")
    for directory in DIRECTORIES:
        create_directory(directory)

    # Create empty files
    print("\nCreating empty placeholder files...")
    for file_path in EMPTY_FILES:
        create_file(file_path) # Creates empty file

    # Create files with content
    print("\nCreating configuration and main script files...")
    create_file(".gitignore", GITIGNORE_CONTENT)
    create_file("requirements.txt", REQUIREMENTS_CONTENT)
    create_file("packages.txt", PACKAGES_CONTENT) # For Hugging Face system deps
    create_file(".env.example", ENV_EXAMPLE_CONTENT)
    create_file("app.py", APP_PY_CONTENT)
    create_file(".env", "# Add your GROQ_API_KEY here\nGROQ_API_KEY=") # Create empty .env

    # Make app.py potentially executable (useful though not strictly necessary)
    try:
        st = os.stat("app.py")
        os.chmod("app.py", st.st_mode | stat.S_IEXEC)
        print("\nMade app.py executable (optional).")
    except Exception as e:
        print(f"\nWarning: Could not make app.py executable: {e}")


    # Go back to the parent directory
    os.chdir("..")

    print(f"\n--- Project '{PROJECT_NAME}' setup complete! ---")
    print("\nNext steps:")
    print(f"1. cd {PROJECT_NAME}")
    print("2. Create a Python virtual environment: python -m venv venv  (or use conda)")
    print("3. Activate the environment:")
    print("   - Linux/macOS: source venv/bin/activate")
    print("   - Windows CMD: .\\venv\\Scripts\\activate.bat")
    print("   - Windows PowerShell: .\\venv\\Scripts\\Activate.ps1")
    print("4. Install dependencies: pip install -r requirements.txt")
    print("5. Copy '.env.example' to '.env' and add your GROQ_API_KEY.")
    print("   - cp .env.example .env (Linux/macOS)")
    print("   - copy .env.example .env (Windows)")
    print("6. Edit '.env' to add your actual API key(s).")
    print("7. If using system packages (like for pymupdf), review 'packages.txt' for Hugging Face deployment.")
    print("8. Start developing: streamlit run app.py")

if __name__ == "__main__":
    main()