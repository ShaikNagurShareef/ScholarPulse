# ScholarPulse: Accelerate Research Understanding and Implementation through AI-Powered Research Paper Analysis and Code Generation Platform

**ScholarPulse** is an AI-powered application designed to help scholars, including bachelor’s and master’s students, understand research papers, answer questions, and generate reliable code implementations. It leverages Groq API, LangChain, LangGraph, and external tools to accelerate research comprehension and experimentation.

Below is a well-structured `README.md` file for your project, based on the provided code from `core/agents.py`, `core/llm_setup.py`, `core/parsing.py`, `core/retrieval.py`, `core/utils.py`, `ui/components.py`, and `ui/views.py`. This README assumes the project is named "ScholarPulse" (based on prompts in the code) and provides an overview, setup instructions, usage details, and additional information tailored to the functionality observed in the code.

```markdown
# ScholarPulse

**ScholarPulse** is an AI-powered research assistant designed to help users understand academic papers by providing question-answering, summarization, and code generation capabilities. Built with LangChain, Streamlit, and Groq's language models, it processes PDFs, URLs (arXiv, DOIs, direct links), and extracts insights tailored to the user's background. The project leverages retrieval-augmented generation (RAG) and vector storage for context-aware responses.

## Features

- **Question Answering**: Ask specific questions about a paper and get concise, context-grounded answers.
- **Paper Summarization**: Generate detailed summaries tailored to your academic or professional background.
- **Code Generation**: Extract algorithms or methods from papers and generate code snippets (currently Python-focused).
- **Flexible Input**: Supports PDF uploads, arXiv IDs/URLs, DOIs, and direct PDF links.
- **User Customization**: Adjusts responses based on user background (e.g., Bachelor's Student, PhD Researcher).
- **Persistent Memory**: Remembers conversation context across sessions (via LangGraph memory).
- **Vector Search**: Uses ChromaDB and SentenceTransformer embeddings for efficient retrieval.

## Project Structure

```
ScholarPulse/
├── core/
│   ├── agents.py        # RAG, summarization, and code generation chains
│   ├── llm_setup.py     # Groq LLM configuration for general and coding tasks
│   ├── parsing.py       # PDF extraction and text chunking from various sources
│   ├── retrieval.py     # Vector store creation and retrieval setup
│   └── utils.py         # Utility functions (e.g., GitHub link extraction, logging)
├── ui/
│   ├── components.py    # Streamlit sidebar UI components
│   └── views.py         # Streamlit views for QA, summary, and code generation
├── app.py               # Main Streamlit application (assumed, not provided)
├── .env                 # Environment variables (e.g., GROQ_API_KEY)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Prerequisites

- Python 3.8+
- Git (for cloning the repository)
- A Groq API key (sign up at [Groq Console](https://console.groq.com/))
- (Optional) Persistent storage directory for ChromaDB (configured in `retrieval.py`)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/ScholarPulse.git
   cd ScholarPulse
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Create a `requirements.txt` file with the following (adjust versions as needed):
   ```
   streamlit
   langchain
   langchain-community
   langchain-groq
   chromadb
   sentence-transformers
   pypdf
   beautifulsoup4
   requests
   python-dotenv
   langgraph
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   UNPAYWALL_EMAIL=your_email@example.com  # Optional, for DOI resolution
   ```
   Replace `your_groq_api_key_here` with your Groq API key.

## Usage

1. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   (Note: `app.py` is assumed to integrate the components and views. If not provided, you'll need to create it—see "Creating `app.py`" below.)

2. **Interact with ScholarPulse**
   - **Sidebar**: Enter a URL (arXiv, DOI, or PDF link), upload a PDF, select your background, and click "Analyze Paper".
   - **Tabs**:
     - **Q&A**: Ask questions about the paper and chat with the assistant.
     - **Summary**: Generate a tailored summary of the paper.
     - **Code**: Request code snippets based on paper content (Python only for now).

3. **Example Inputs**
   - URL: `https://arxiv.org/abs/1706.03762` (Attention is All You Need)
   - DOI: `10.1109/CVPR.2016.90` (ResNet paper, may require open-access link)
   - PDF: Upload a local research paper PDF.

## Configuration Details

- **LLM Models**: Uses Groq's `meta-llama/llama-4-maverick-17b-128e-instruct` for general tasks and `qwen-2.5-coder-32b` for coding (configurable in `llm_setup.py`).
- **Embedding Model**: SentenceTransformers `all-MiniLM-L6-v2` for vector search (configurable in `retrieval.py`).
- **Chunking**: Text split into 1000-character chunks with 100-character overlap (configurable in `parsing.py`).
- **Logging**: Comprehensive logging across all modules with timestamp, name, level, and message.

## Limitations

- **Code Generation**: Experimental, Python-only, and may require manual review.
- **Long Papers**: Summarization truncates at ~20,000 characters (configurable in `views.py`).
- **Persistence**: Vector store is in-memory by default; set `PERSIST_DIRECTORY` in `retrieval.py` for persistence.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License (or specify your preferred license).

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain), [Streamlit](https://streamlit.io/), and [Groq](https://groq.com/).
- Embedding models from [SentenceTransformers](https://huggingface.co/sentence-transformers).
- Inspired by the need to make research papers more accessible
