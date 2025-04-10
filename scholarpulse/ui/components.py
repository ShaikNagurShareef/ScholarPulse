# Filename: ui/components.py

import streamlit as st
import logging
from core.utils import get_logger

# Initialize logger
logger = get_logger(__name__)

def display_sidebar():
    """
    Displays the sidebar components for input and user background selection.

    Returns:
        dict: A dictionary containing the *values* from the sidebar widgets.
              Keys: 'source_input', 'uploaded_file', 'user_background', 'process_clicked'
    """
    sidebar_values = {}

    with st.sidebar:
        st.header("📝 Input Paper")
        sidebar_values['source_input'] = st.text_input(
            "Enter arXiv /Open Source URL",
            key="source_input_key" # Add key for stability
        )
        sidebar_values['uploaded_file'] = st.file_uploader(
            "OR Upload PDF",
            type=['pdf'],
            key="uploaded_file_key"
        )

        st.markdown("---") # Visual separator

        st.header("👤 Your Background")
        sidebar_values['user_background'] = st.selectbox(
            "Select your background level:",
            ("Bachelor's Student", "Master's Student", "PhD Student/Researcher", "Industry Professional", "Curious Learner"),
            key="user_background_key",
            index=1 # Default to Master's Student
        )

        st.markdown("---")

        sidebar_values['process_clicked'] = st.button(
            "Analyze Paper ✨",
            key="process_button_key"
        )

        # Optional: Add disclaimer or info
        # st.markdown("---")
        st.caption("Ensure input links point directly to PDFs where possible for best results.")

    return sidebar_values

# You can add more reusable components here later, e.g.:
# def display_chat_interface(...)
# def display_summary(...)
# def display_code(...)