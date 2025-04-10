# Filename: core/retrieval.py

import os
import chromadb # Ensure chromadb is installed
# If using sentence-transformers locally:
from langchain_community.embeddings import SentenceTransformerEmbeddings
# Alternatively, configure for hosted embeddings if preferred (e.g., OpenAI, HuggingFace Inference API)
# from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.schema import Document # Used if splitting with create_documents
from langchain.vectorstores.base import VectorStoreRetriever # More explicit import
import logging
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Configuration ---
# Model for creating embeddings. Choose one suitable for your task and resources.
# 'all-MiniLM-L6-v2' is small and fast, good for CPU.
# 'all-mpnet-base-v2' is generally better quality but larger.
# Check Sentence Transformers documentation for more models.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Directory to persist ChromaDB data (optional, can run in-memory)
# For Hugging Face Spaces, consider if you need persistence across restarts.
# In-memory is simpler but means re-indexing on each app start/restart.
PERSIST_DIRECTORY = None # Set to a path like "./chroma_db_store" to persist
# PERSIST_DIRECTORY = "chroma_db_store" # Example path for persistence

# Number of relevant chunks to retrieve
DEFAULT_K = 4

# --- Embedding Function ---

def get_embedding_function():
    """
    Initializes and returns the embedding function.
    """
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        # Check if CUDA is available for GPU acceleration (optional but recommended if possible)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # logger.info(f"Using device: {device}")
        embedding_function = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            # model_kwargs={'device': device}, # Uncomment if using torch and device detection
            encode_kwargs={'normalize_embeddings': True} # Normalize for better similarity search
        )
        logger.info("Embedding model loaded successfully.")
        return embedding_function
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        # Potentially fall back to a different model or raise the error
        raise ValueError(f"Could not load embedding model: {EMBEDDING_MODEL_NAME}") from e

    # --- Alternative: OpenAI Embeddings (Requires OPENAI_API_KEY) ---
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY not found for OpenAI embeddings.")
    # return OpenAIEmbeddings(openai_api_key=api_key)

    # --- Alternative: HuggingFace Inference API Embeddings (Requires HF_TOKEN) ---
    # api_key = os.getenv("HF_TOKEN")
    # if not api_key:
    #     raise ValueError("HF_TOKEN not found for HuggingFace Inference API embeddings.")
    # return HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="sentence-transformers/all-MiniLM-L6-v2") # Example


# --- Vector Store Management ---

def create_vector_store(text_chunks, embedding_function, collection_name="paper_collection"):
    """
    Creates a Chroma vector store from text chunks.

    Args:
        text_chunks (list[str]): List of text segments from the paper.
        embedding_function: The embedding function instance to use.
        collection_name (str): Name for the Chroma collection (useful if persisting).

    Returns:
        Chroma: An instance of the Chroma vector store populated with the chunks,
                or None if creation fails.
    """
    if not text_chunks:
        logger.error("Error: No text chunks provided to create vector store.")
        return None

    logger.info(f"Creating vector store with {len(text_chunks)} chunks...")
    try:
        # If using LangChain's Document objects (e.g., from text_splitter.create_documents):
        # vector_store = Chroma.from_documents(
        #     documents=text_chunks, # Assuming text_chunks are Document objects
        #     embedding=embedding_function,
        #     collection_name=collection_name,
        #     persist_directory=PERSIST_DIRECTORY # Will be None if not persisting
        # )

        # If text_chunks is just a list of strings:
        vector_store = Chroma.from_texts(
            texts=text_chunks,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=PERSIST_DIRECTORY
        )

        if PERSIST_DIRECTORY:
            logger.info(f"Persisting vector store to: {PERSIST_DIRECTORY}/{collection_name}")
            vector_store.persist() # Ensure data is saved if persist_directory is set

        logger.info("Vector store created successfully.")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None


def load_vector_store(embedding_function, collection_name="paper_collection"):
    """
    Loads an existing persisted Chroma vector store.

    Args:
        embedding_function: The embedding function instance (must match the one used for creation).
        collection_name (str): Name of the Chroma collection to load.

    Returns:
        Chroma: An instance of the loaded Chroma vector store, or None if loading fails or
                persistence is not configured/directory doesn't exist.
    """
    if not PERSIST_DIRECTORY:
        logger.warning("Persistence directory not configured. Cannot load vector store.")
        return None

    if not os.path.exists(PERSIST_DIRECTORY):
        logger.warning(f"Persistence directory '{PERSIST_DIRECTORY}' not found. Cannot load vector store.")
        return None

    logger.info(f"Attempting to load vector store from: {PERSIST_DIRECTORY}/{collection_name}")
    try:
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
        logger.info("Vector store loaded successfully.")
        # Verify collection exists (optional check)
        # if not vector_store._client.get_collection(collection_name):
        #      logger.warning(f"Warning: Collection '{collection_name}' not found in loaded store.")
        #      return None # Or handle as needed
        return vector_store
    except Exception as e:
        # ChromaDB might raise specific exceptions, catch broadly for now
        logger.error(f"Error loading vector store from {PERSIST_DIRECTORY}: {e}")
        return None


# --- Retriever Creation ---

def get_retriever_from_store(vector_store, search_type="similarity", k=DEFAULT_K):
    """
    Creates a retriever object from a vector store instance.

    Args:
        vector_store (Chroma): The vector store instance.
        search_type (str): Type of search ('similarity', 'mmr', etc.).
        k (int): Number of documents to retrieve.

    Returns:
        VectorStoreRetriever: A retriever instance, or None if input is invalid.
    """
    if not vector_store:
        logger.error("Error: Invalid vector store provided.")
        return None

    logger.info(f"Creating retriever (search_type='{search_type}', k={k})...")
    try:
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k}
        )
        logger.info("Retriever created successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        return None

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    # Dummy data for testing
    sample_chunks = [
        "The Transformer architecture relies heavily on self-attention mechanisms.",
        "Self-attention allows the model to weigh the importance of different words in the input sequence.",
        "Multi-head attention runs the attention mechanism multiple times in parallel.",
        "Positional encodings are added to give the model information about word order.",
        "The encoder stack processes the input sequence.",
        "The decoder stack generates the output sequence, attending to the encoder output."
    ]

    logger.info("\n--- Testing Embedding Function ---")
    try:
        embed_func = get_embedding_function()
    except ValueError as e:
        logger.error(str(e))
        embed_func = None

    if embed_func:
        logger.info("\n--- Testing Vector Store Creation (In-Memory) ---")
        # Use a different collection name for testing to avoid conflicts if persisting
        test_collection_name = "test_collection"
        # Ensure clean state for in-memory test if PERSIST_DIRECTORY is set globally
        original_persist_dir = PERSIST_DIRECTORY
        PERSIST_DIRECTORY = None # Force in-memory for this test section

        vs = create_vector_store(sample_chunks, embed_func, collection_name=test_collection_name)

        if vs:
            logger.info("\n--- Testing Retriever Creation ---")
            retriever = get_retriever_from_store(vs, k=2)

            if retriever:
                logger.info("\n--- Testing Retrieval ---")
                query = "What is multi-head attention?"
                try:
                    results = retriever.invoke(query) # Use invoke for LCEL compatibility
                    logger.info(f"Query: '{query}'")
                    logger.info("Results:")
                    for i, doc in enumerate(results):
                        logger.info(f"  {i+1}. {doc.page_content}") # Assuming results are Document objects
                except Exception as e:
                    logger.error(f"Error during retrieval test: {e}")
            else:
                logger.error("Skipping retrieval test as retriever creation failed.")

        else:
            logger.error("Skipping retriever/retrieval tests as vector store creation failed.")

        # Restore original persist directory setting if needed
        PERSIST_DIRECTORY = original_persist_dir

        # --- Optional: Test Persistence ---
        if PERSIST_DIRECTORY:
            logger.info(f"\n--- Testing Vector Store Persistence (Directory: {PERSIST_DIRECTORY}) ---")
            persist_collection_name = "persistent_test"
            vs_persist = create_vector_store(sample_chunks, embed_func, collection_name=persist_collection_name)
            if vs_persist:
                logger.info("Simulating app restart - attempting to load store...")
                del vs_persist # Remove from memory to force loading
                loaded_vs = load_vector_store(embed_func, collection_name=persist_collection_name)
                if loaded_vs:
                    logger.info("Successfully loaded persisted store.")
                    # You could add a retrieval test here on the loaded_vs
                else:
                    logger.error("Failed to load persisted store.")
            else:
                logger.error("Failed to create store for persistence test.")
    else:
        logger.error("\nSkipping vector store and retriever tests as embedding function failed.")