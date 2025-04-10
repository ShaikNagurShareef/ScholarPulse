# Filename: core/parsing.py

import os
import re
import io
import requests
from urllib.parse import urlparse
from pypdf import PdfReader # Using pypdf, ensure it's in requirements.txt
# from PyPDF2 import PdfReader # Alternative library, choose one
from bs4 import BeautifulSoup # For trying to find PDF links on HTML pages

from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Configuration ---
# Adjust chunk size and overlap based on model context window and desired granularity
# Smaller chunks are better for RAG but increase the number of chunks.
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 100

# --- Helper Functions ---

def is_valid_url(url):
    """Checks if the provided string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def resolve_arxiv_url(arxiv_id_or_url):
    """Resolves an arXiv ID or URL to its PDF download link."""
    # Regex to find arXiv IDs (e.g., 2303.10130, hep-th/0203034, etc.)
    arxiv_regex = r'(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?\/\d{7})(v\d+)?'
    match = re.search(arxiv_regex, arxiv_id_or_url)
    if match:
        arxiv_id = match.group(1)
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # Handle cases where the full URL might already be a PDF link
    if arxiv_id_or_url.endswith('.pdf') and 'arxiv.org' in arxiv_id_or_url:
        return arxiv_id_or_url
    # Handle cases where it's an abstract page URL
    if 'arxiv.org/abs/' in arxiv_id_or_url:
        pdf_url = arxiv_id_or_url.replace('/abs/', '/pdf/') + ".pdf"
        return pdf_url
    return None # Return None if no valid arXiv ID or format found

def resolve_doi_url(doi):
    """Tries to resolve a DOI to a publicly accessible PDF URL using Unpaywall (best effort)."""
    # Basic DOI validation (doesn't guarantee existence)
    if not doi or not doi.startswith('10.'):
        return None

    # Using Unpaywall API - requires an email for polite use
    # In a production app, handle rate limits and errors more robustly.
    try:
        # Replace YOUR_EMAIL@example.com or consider making it an env variable
        email = os.getenv("UNPAYWALL_EMAIL", "scholarpulse.app@example.com")
        api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        response = requests.get(api_url, timeout=15) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        pdf_url_info = data.get('best_oa_location')
        if pdf_url_info and pdf_url_info.get('url_for_pdf'):
            logger.info(f"Found OA PDF for DOI {doi} via Unpaywall: {pdf_url_info.get('url_for_pdf')}")
            return pdf_url_info.get('url_for_pdf')
        else:
            # Fallback: Construct potential publisher link (less reliable for direct PDF)
            # This often leads to an HTML page, not a direct PDF.
            logger.info(f"No direct OA PDF found via Unpaywall for DOI {doi}. Trying CrossRef...")
            # Attempt direct resolution via dx.doi.org (might hit paywall/landing page)
            # return f"https://doi.org/{doi}" # Usuallygoes to landing page
            return None # Indicate direct PDF not found easily
    except requests.exceptions.RequestException as e:
        logger.error(f"Error resolving DOI {doi} via Unpaywall: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error resolving DOI {doi}: {e}")
        return None


def fetch_url_content(url):
    """Fetches content from a URL, prioritizing PDF."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 ScholarPulseBot/1.0'
    }
    try:
        response = requests.get(url, headers=headers, timeout=20, stream=True) # Use stream=True for potentially large PDFs
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()

        if 'application/pdf' in content_type:
            logger.info(f"Identified direct PDF content from URL: {url}")
            # Read content into memory - careful with very large files
            # For extremely large files, saving to disk might be better
            pdf_content = io.BytesIO(response.content)
            return pdf_content, 'pdf'
        elif 'text/html' in content_type:
            logger.info(f"URL {url} returned HTML. Attempting to find PDF link...")
            # Basic attempt to find a PDF link on the page (highly heuristic)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for anchor tags linking to '.pdf' files
            pdf_link_tag = soup.find('a', href=lambda href: href and href.lower().endswith('.pdf'))
            if pdf_link_tag:
                pdf_url = requests.compat.urljoin(url, pdf_link_tag['href']) # Resolve relative links
                logger.info(f"Found potential PDF link on page: {pdf_url}")
                # Fetch the found PDF link
                return fetch_url_content(pdf_url) # Recursive call to handle the new URL

            logger.warning(f"Could not find a direct PDF link on HTML page: {url}")
            return None, 'html' # Indicate HTML content was found, but no PDF link
        else:
            logger.warning(f"Unsupported content type '{content_type}' at URL: {url}")
            return None, content_type

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching URL: {url}")
        return None, 'error_timeout'
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None, 'error_request'
    except Exception as e:
        logger.error(f"Unexpected error fetching URL {url}: {e}")
        return None, 'error_unexpected'


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file object (BytesIO or file pointer)."""
    text = ""
    try:
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        logger.info(f"Reading {num_pages} pages from PDF...")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n" # Add separator between pages
                else:
                    logger.warning(f"Warning: No text extracted from page {i+1}.")
            except Exception as page_error:
                # Sometimes specific pages can cause errors in extraction
                logger.warning(f"Warning: Could not extract text from page {i+1}: {page_error}")
        logger.info("Finished extracting text from PDF.")
        # Basic cleanup (optional, more advanced cleaning might be needed)
        text = re.sub(r'\s+', ' ', text).strip() # Replace multiple whitespace with single space
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return None

def split_text(text):
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    if not text:
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_CHUNK_SIZE,
            chunk_overlap=TEXT_CHUNK_OVERLAP,
            length_function=len,
            # separators=["\n\n", "\n", ".", ",", " ", ""] # Default separators are usually good
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks.")
        # Optional: Add metadata generation here if needed per chunk later
        # docs = text_splitter.create_documents([text]) # Creates Document objects with metadata
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return [] # Return empty list on error


# --- Main Processing Function ---

def process_input_source(source):
    """
    Processes the input source (file upload, URL, DOI, arXiv ID)
    and returns the extracted text chunks.

    Args:
        source (str or file-like object):
            - If string: Assumed to be URL, DOI, or arXiv ID.
            - If file-like object (from Streamlit upload): Assumed to be PDF.

    Returns:
        list[str] or None: A list of text chunks if successful, otherwise None.
    """
    extracted_text = None

    if hasattr(source, 'read'): # Check if it's a file-like object (Streamlit upload)
        logger.info("Processing uploaded file...")
        # Assume it's PDF for now, could add type checking
        if source.type == "application/pdf":
             # Ensure we read the file content into BytesIO if needed by PdfReader
            pdf_content = io.BytesIO(source.read())
            extracted_text = extract_text_from_pdf(pdf_content)
        else:
            logger.error(f"Unsupported file type uploaded: {source.type}")
            return None
    elif isinstance(source, str):
        source = source.strip()
        logger.info(f"Processing input string: {source}")
        pdf_url = None
        # 1. Check for arXiv
        pdf_url = resolve_arxiv_url(source)
        if pdf_url:
            logger.info(f"Resolved arXiv source to PDF URL: {pdf_url}")
        else:
            # 2. Check for DOI
            # Basic DOI pattern check (could be more robust)
            if re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', source, re.IGNORECASE):
                 pdf_url = resolve_doi_url(source)
                 if pdf_url:
                     logger.info(f"Resolved DOI to PDF URL: {pdf_url}")
                 else:
                     logger.warning(f"Could not resolve DOI '{source}' to a direct PDF URL.")
                     # Optionally, try fetching the landing page anyway?
                     # pdf_url = f"https://doi.org/{source}" # This likely points to HTML
                     return None # Fail if direct PDF not found for DOI

            # 3. Check if it's a standard URL
            elif is_valid_url(source):
                 pdf_url = source # Assume it might be a direct PDF link or leads to one
                 logger.info(f"Processing as standard URL: {pdf_url}")
            else:
                logger.error(f"Input string '{source}' is not a valid URL, arXiv ID, or DOI.")
                return None

        # Fetch content if we have a URL
        if pdf_url:
            content, content_type = fetch_url_content(pdf_url)
            if content and content_type == 'pdf':
                extracted_text = extract_text_from_pdf(content)
            elif content_type == 'html':
                logger.warning("URL pointed to an HTML page, and no direct PDF link was found.")
                return None # Indicate failure, as we need PDF content
            elif content is None:
                logger.error(f"Failed to fetch or process content from URL: {pdf_url} (Type: {content_type})")
                return None

    else:
        logger.error(f"Unsupported input type: {type(source)}")
        return None

    # Split the extracted text if successful
    if extracted_text:
        chunks = split_text(extracted_text)
        if chunks:
            logger.info(f"Successfully extracted and chunked text from source.")
            return chunks
        else:
            logger.error("Text extraction succeeded, but splitting failed.")
            return None
    else:
        logger.error("Text extraction failed.")
        return None


# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    # Test cases (replace with actual URLs/DOIs/PDF paths for real testing)
    test_sources = [
        "https://arxiv.org/abs/1706.03762", # Attention is All You Need (Abstract page)
        "1706.03762v5",                   # Attention is All You Need (ID with version)
        "https://arxiv.org/pdf/2303.10130.pdf", # Direct PDF link
        "10.1109/CVPR.2016.90",            # ResNet DOI (likely paywalled/HTML) -> Will likely fail find OA PDF
        "https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf", # Direct link to ResNet PDF
        # Add path to a local PDF file for testing uploads:
        # "/path/to/your/local/paper.pdf" # Uncomment and replace path
    ]

    for source in test_sources:
        logger.info(f"--- Testing Source: {source} ---")
        if os.path.exists(source) and source.endswith(".pdf"): # Simulate file upload
            logger.info("(Simulating file upload)")
            try:
                with open(source, "rb") as f:
                    # Create a mock Streamlit UploadedFile object
                    class MockUploadedFile:
                        def __init__(self, file, name, type):
                            self._file = io.BytesIO(file.read())
                            self.name = name
                            self.type = type
                        def read(self):
                            return self._file.read()
                        def seek(self, pos): # Important for rereading if needed
                            self._file.seek(pos)

                    mock_file = MockUploadedFile(f, os.path.basename(source), "application/pdf")
                    result = process_input_source(mock_file)
            except FileNotFoundError:
                logger.error(f"Local test PDF not found: {source}")
                result = None
        else: # Treat as string input (URL, DOI, arXiv ID)
            result = process_input_source(source)

        if result:
            logger.info(f"Successfully processed. Number of chunks: {len(result)}")
            # logger.info(f"First chunk: {result[0][:200]}...") # Optionally log first chunk
        else:
            logger.error(f"Failed to process source: {source}")
        logger.info("-" * 25)