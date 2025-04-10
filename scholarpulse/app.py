# Filename: app.py

import streamlit as st
from dotenv import load_dotenv
import os
import sys
import time
import logging
from core.utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Environment and Path Setup ---
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))

# --- Core Imports ---
from core.parsing import process_input_source
from core.retrieval import get_embedding_function, create_vector_store, get_retriever_from_store
from core.agents import create_agentic_rag_chain, create_rag_chain, create_summarization_chain, create_code_generation_chain
from core.llm_setup import get_llm # Keep for early key check

# --- UI Imports ---
from ui.components import display_sidebar
# Import the view functions
from ui.views import render_qa_view, render_summary_view, render_code_view


# --- LangChain Imports ---
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Configuration ---
st.set_page_config(
    page_title="ScholarPulse",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 ScholarPulse: AI Research Assistant")
st.caption("Your intelligent companion for understanding research papers.")

# --- Session State Initialization ---
# (Keep the session state initialization as it was)
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
# ... (keep all other session state initializations) ...
if 'vector_store' not in st.session_state: st.session_state.vector_store = None
if 'retriever' not in st.session_state: st.session_state.retriever = None
if 'rag_chain' not in st.session_state: st.session_state.rag_chain = None
if 'summarize_chain' not in st.session_state: st.session_state.summarize_chain = None
if 'code_chain' not in st.session_state: st.session_state.code_chain = None
if 'paper_chunks' not in st.session_state: st.session_state.paper_chunks = None
if 'user_background' not in st.session_state: st.session_state.user_background = "Master's Student"


# --- Check for API Key Early ---
api_key_exists = bool(os.getenv("GROQ_API_KEY"))
if not api_key_exists:
    st.error("🚨 GROQ_API_KEY not found! Please set it in your .env file.")
    st.stop()

# --- Sidebar ---
sidebar_values = display_sidebar()
st.session_state.user_background = sidebar_values['user_background']

# --- Main Processing Logic ---
# (Keep the processing logic block exactly as it was in Step 5b)
# This block populates st.session_state with the necessary chains, retriever, etc.
if sidebar_values['process_clicked']:
    # Reset state for new analysis
    st.session_state.processing_complete = False
    st.session_state.error_message = None
    # ... (reset all relevant session state variables) ...
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.session_state.summarize_chain = None
    st.session_state.code_chain = None
    st.session_state.paper_chunks = None

    input_source = None
    if sidebar_values['uploaded_file']:
        input_source = sidebar_values['uploaded_file']
        logger.info(f"Received file upload: {input_source.name}")
    elif sidebar_values['source_input']:
        input_source = sidebar_values['source_input']
        logger.info(f"Received text input: {input_source}")
    else:
        st.warning("Please upload a PDF or enter a URL/DOI.")
        st.stop() # Stop processing if no input

    if input_source:
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty() # Placeholder for status updates
        try:
            # --- Start Processing Pipeline ---
            status_text.info("1/5 - Processing input source...")
            progress_bar.progress(10, text="Processing input source...")
            chunks = process_input_source(input_source)
            if not chunks: raise ValueError("Failed to parse or extract text.")
            st.session_state.paper_chunks = chunks
            progress_bar.progress(30, text="Text extracted and chunked.")
            time.sleep(0.5)

            status_text.info("2/5 - Initializing embedding model...")
            progress_bar.progress(40, text="Initializing embedding model...")
            embedding_function = get_embedding_function()
            progress_bar.progress(50, text="Embedding model ready.")
            time.sleep(0.5)

            status_text.info("3/5 - Creating vector store...")
            progress_bar.progress(60, text="Creating vector store...")
            st.session_state.vector_store = create_vector_store(chunks, embedding_function)
            if not st.session_state.vector_store: raise ValueError("Failed to create vector store.")
            progress_bar.progress(75, text="Vector store created.")
            time.sleep(0.5)

            status_text.info("4/5 - Setting up retriever and agents...")
            progress_bar.progress(85, text="Setting up retriever and agents...")
            st.session_state.retriever = get_retriever_from_store(st.session_state.vector_store)
            if not st.session_state.retriever: raise ValueError("Failed to create retriever.")


            st.session_state.rag_chain = create_rag_chain(st.session_state.retriever, st.session_state.user_background)
            st.session_state.agentic_rag_chain = create_agentic_rag_chain(st.session_state.retriever, st.session_state.user_background)
            st.session_state.summarize_chain = create_summarization_chain(st.session_state.user_background)
            st.session_state.code_chain = create_code_generation_chain(st.session_state.user_background)
            progress_bar.progress(95, text="Agents ready.")
            time.sleep(0.5)

            status_text.success("5/5 - Processing Complete!")
            progress_bar.progress(100, text="Processing Complete!")
            st.session_state.processing_complete = True
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            # --- End Processing Pipeline ---
        except Exception as e:
            st.session_state.error_message = f"An error occurred: {e}"
            st.error(st.session_state.error_message)
            logger.error(f"Error during processing pipeline: {e}")
            status_text.empty()
            progress_bar.empty()
            st.session_state.processing_complete = False


# --- Main Area for Displaying Results ---
if st.session_state.processing_complete:
    st.success("✅ Paper processed! Select a tab below.")
    st.markdown("---")

    # --- Use Tabs and Call View Functions ---
    tab1, tab2, tab3 = st.tabs(["❓ Q&A", "📄 Summary", "💻 Code"])

    with tab1:
        # Call the Q&A view function, passing the required chain from session state
        render_qa_view(st.session_state.get('rag_chain')) # Use .get for safety

    with tab2:
        # Call the Summary view function
        render_summary_view(
            st.session_state.get('summarize_chain'),
            st.session_state.get('paper_chunks'),
            st.session_state.user_background # Pass background directly 
        )

    with tab3:
        # Call the Code view function
        render_code_view(
            st.session_state.get('code_chain'),
            st.session_state.get('retriever'),
            st.session_state.user_background # Pass background directly
        )

elif st.session_state.error_message:
    st.error(f"Processing failed: {st.session_state.error_message}")
else:
    st.info("Upload a paper or enter a source in the sidebar and click 'Analyze Paper'.")

# --- Optional Footer ---
st.markdown("---")
st.caption("Powered by LangChain, Groq, and Streamlit. Handle AI-generated content with care.")