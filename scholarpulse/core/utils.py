# Filename: core/utils.py

import re
import requests # Keep requests if needed for other utils, like DOI resolution
import logging

# Initialize logger at the module level
logger = None  # Will be initialized via get_logger when needed

# --- Constants ---
# Regex to find GitHub URLs (might need refinement for edge cases)
# Looks for standard github.com URLs optionally preceded by http(s):// and www.
# Captures the owner/repo part.
GITHUB_URL_PATTERN = re.compile(r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)')

# --- Utility Functions ---

def find_github_links_in_text(text):
    """
    Scans text for potential GitHub repository links.

    Args:
        text (str): The text content to scan (e.g., the full paper text).

    Returns:
        list[str]: A list of unique, formatted GitSub URLs found in the text.
                   Returns an empty list if no links are found or text is empty.
    """
    global logger
    if logger is None:
        logger = get_logger(__name__)

    if not text:
        return []

    found_links = set() # Use a set to automatically handle duplicates

    # Find all matches using the regex
    matches = GITHUB_URL_PATTERN.findall(text)

    for repo_path in matches:
        # Format into a standard https URL
        full_url = f"https://github.com/{repo_path}"
        # Basic cleanup: remove trailing characters like periods or commas if accidentally captured
        # (though the regex tries to avoid this, sometimes formatting is tricky)
        full_url = re.sub(r'[.,\s]*$', '', full_url)
        found_links.add(full_url)

    if found_links:
        logger.info(f"Found {len(found_links)} potential GitHub link(s): {list(found_links)}")
    else:
        logger.info("No GitHub links found in the provided text.")

    return sorted(list(found_links)) # Return sorted list

# --- Add other utility functions below as needed ---
# E.g., DOI resolution helpers, API wrappers, etc.

"""Logging utility for the Agentic RAG system."""

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    # Ensure logger is initialized for the test section
    logger = get_logger(__name__)

    sample_text_with_links = """
    This paper introduces a novel technique. The code is available at
    https://github.com/example-user/cool-repo. We also reference
    work from github.com/another-org/utility-lib (see Section 4).
    Check www.github.com/example-user/cool-repo for updates.
    An older version was at http://github.com/legacy/project.git.
    This is not a link: github. com / spaced / out.
    """
    sample_text_without_links = """
    This document describes a process without any external code repositories mentioned.
    Standard libraries were used.
    """

    logger.info("\n--- Testing with links ---")
    links = find_github_links_in_text(sample_text_with_links)
    logger.info(f"Result: {links}")
    assert "https://github.com/example-user/cool-repo" in links
    assert "https://github.com/another-org/utility-lib" in links
    assert "https://github.com/legacy/project" in links # .git should ideally be handled if pattern allows
    assert len(links) == 3 # Check for uniqueness

    logger.info("\n--- Testing without links ---")
    links = find_github_links_in_text(sample_text_without_links)
    logger.info(f"Result: {links}")
    assert len(links) == 0

    logger.info("\n--- Testing with empty text ---")
    links = find_github_links_in_text("")
    logger.info(f"Result: {links}")
    assert len(links) == 0