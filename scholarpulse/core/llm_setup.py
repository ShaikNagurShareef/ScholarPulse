# Filename: core/llm_setup.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import logging
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Ensure environment variables are loaded
load_dotenv()

# --- Configuration ---
# Define model names for different purposes
# Check Groq console/docs for available models: https://console.groq.com/docs/models
# Example: Llama 3 8B is fast and capable for general tasks.
GENERAL_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
# Example: Mixtral is often good for reasoning and coding tasks.
CODING_MODEL_NAME = "qwen-2.5-coder-32b"
# Or use the one you had: "meta-llama/llama-4-maverick-17b-128e-instruct"
# GENERAL_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- Helper to get API Key ---
def _get_groq_api_key():
    """Retrieves the Groq API key, raising an error if not found."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    return api_key

# --- LLM Getters ---

def get_llm():
    """
    Initializes and returns the Groq LLM instance configured for general tasks
    (Q&A, Summarization).

    Returns:
        ChatGroq: An instance of the ChatGroq LLM for general use.
    """
    api_key = _get_groq_api_key()
    llm = ChatGroq(
        temperature=0.1, # Lower temperature for factual, consistent outputs
        groq_api_key=api_key,
        model_name=GENERAL_MODEL_NAME
    )
    logger.info(f"Initialized General LLM: {GENERAL_MODEL_NAME}")
    return llm

def get_codellm():
    """
    Initializes and returns the Groq LLM instance configured specifically for
    code generation tasks.

    Returns:
        ChatGroq: An instance of the ChatGroq LLM for coding use.
    """
    api_key = _get_groq_api_key()
    llm = ChatGroq(
        # Slightly higher temperature might allow more creative coding solutions,
        # but keep it low to avoid hallucinated code. Adjust based on results.
        temperature=0.2,
        groq_api_key=api_key,
        model_name=CODING_MODEL_NAME
    )
    logger.info(f"Initialized Coding LLM: {CODING_MODEL_NAME}")
    return llm

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    logger.info("--- Testing LLM Initialization ---")
    try:
        general_llm_instance = get_llm()
        logger.info(f"Successfully initialized General LLM: {type(general_llm_instance)}")
        logger.info(f"Using model: {general_llm_instance.model_name}")

        logger.info("-" * 20)

        coding_llm_instance = get_codellm()
        logger.info(f"Successfully initialized Coding LLM: {type(coding_llm_instance)}")
        logger.info(f"Using model: {coding_llm_instance.model_name}")

    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")