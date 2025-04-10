# Filename: ui/views.py

import streamlit as st
# import time # Can be used for simulating delays or just part of debugging
from langchain_core.messages import HumanMessage, AIMessage
import logging
from core.utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# def render_qa_view(rag_chain):
#     st.header("❓ Ask Questions about the Paper")
#     st.caption("Ask specific questions and get answers grounded in the paper's content.")

#     if "qa_messages" not in st.session_state:
#         st.session_state.qa_messages = []

#     for message in st.session_state.qa_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("What is your question?"):
#         st.session_state.qa_messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         if rag_chain:
#             with st.chat_message("assistant"):
#                 message_placeholder = st.empty()
#                 full_response = ""
#                 try:
#                     with st.spinner("Thinking..."):
#                         # Pass user_background from session state along with input and context
#                         response = rag_chain.invoke({
#                             "input": prompt,
#                             "user_background": st.session_state.user_background
#                         })

#                     if isinstance(response, dict) and 'answer' in response:
#                         full_response = response['answer']
#                         message_placeholder.markdown(full_response)
#                     else:
#                         full_response = str(response)
#                         message_placeholder.write(full_response)

#                 except Exception as e:
#                     full_response = f"Sorry, I encountered an error: {e}"
#                     message_placeholder.error(full_response)

#             st.session_state.qa_messages.append({"role": "assistant", "content": full_response})
#         else:
#             st.error("RAG chain is not available. Please process a paper first.")


def render_qa_view(rag_chain):
    st.header("❓ Ask Questions about the Paper")
    st.caption("Chat with the assistant to get answers based on the paper.")

    if "qa_thread_id" not in st.session_state:
        st.session_state.qa_thread_id = "qa_thread_1"
    config = {"configurable": {"thread_id": st.session_state.qa_thread_id}}

    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []
        st.session_state.qa_messages.append({
            "role": "assistant",
            "content": "Hi! I'm ScholarPulse, your AI Research Assistant here to boost your understanding of the uploaded paper. Ask me anything, or check the Summary and Code tabs for more!"
        })

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.qa_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.qa_messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    with st.spinner("Thinking..."):
                        agentic_chain = st.session_state.get("agentic_rag_chain")
                        if agentic_chain:
                            messages = [HumanMessage(content=prompt)]
                            logger.info(f"Invoking with message: {prompt}")
                            response = agentic_chain.invoke({"messages": messages}, config=config)
                            logger.info(f"Response messages: {[m.type for m in response['messages']]}")
                            full_response = response["messages"][-1].content
                            if not full_response.strip():
                                full_response = "I couldn’t generate a response. Please try again or rephrase your question."
                            message_placeholder.markdown(full_response)
                        else:
                            full_response = "Agentic RAG chain not available."
                            message_placeholder.error(full_response)
                except Exception as e:
                    full_response = f"Sorry, I encountered an error: {e}"
                    message_placeholder.error(full_response)

        st.session_state.qa_messages.append({"role": "assistant", "content": full_response})


def render_summary_view(summarize_chain, paper_chunks, user_background):
    """
    Renders the Summary generation interface.

    Args:
        summarize_chain: The initialized LangChain summarization runnable.
        paper_chunks (list[str]): List of text chunks from the paper.
        user_background (str): User's selected background.
    """
    st.header("📄 Paper Summary")
    st.caption("Get a concise overview of the paper tailored to your background.")

    if st.button("Generate Summary ✨", key="summarize_button_view"):
        if summarize_chain and paper_chunks:
            with st.spinner("Generating summary... This may take a moment."):
                try:
                    # Join chunks for the 'stuff' summarizer
                    # TODO: Add logic for map_reduce if implemented later
                    full_text = "\n\n".join(paper_chunks)

                    # Simple length check (replace with proper token counting if needed)
                    # Use a reasonable estimate for context window limits (e.g., ~4k tokens for llama3-8b, ~32k for mixtral)
                    # A character count is a very rough proxy. 1 token ~= 4 chars average.
                    # llama3-8b 8192 tokens * 3 chars/token ~= 24k chars
                    # mixtral 32768 tokens * 3 chars/token ~= 98k chars
                    # Let's use a safer limit for the simpler model
                    max_chars = 20000
                    if len(full_text) > max_chars:
                        st.warning(f"Paper text is quite long ({len(full_text)} chars). Summary might be based on the first ~{max_chars} characters for efficiency.")
                        input_text_for_summary = full_text[:max_chars]
                    else:
                        input_text_for_summary = full_text

                    # Invoke the chain. Ensure the input dict keys match the prompt template.
                    summary = summarize_chain.invoke({
                        "text": input_text_for_summary,
                        "user_background": user_background
                    })
                    st.markdown(summary)
                    st.success("Summary generated!")

                except Exception as e:
                    st.error(f"Error generating summary: {e}")
                    logger.error(f"Summarization error: {e}")
        elif not paper_chunks:
             st.warning("Paper text not available. Please process a paper first.")
        else:
            st.error("Summarization chain is not available.")
    else:
        st.info("Click the button above to generate a summary.")


def render_code_view(code_chain, retriever, user_background):
    """
    Renders the Code Generation interface.

    Args:
        code_chain: The initialized LangChain code generation runnable.
        retriever: The retriever to fetch context relevant to the code request.
        user_background (str): User's selected background.
    """
    st.header("💻 Code Generation")
    st.caption("Generate code snippets based on the paper's content (experimental).")

    code_request = st.text_area(
        "Describe the algorithm, method, or concept you want code for:",
        key="code_request_input_view",
        height=100,
        placeholder="e.g., Implement the data preprocessing steps mentioned in Section 2.1"
    )

    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Language",
            ("Python"),
            key="code_lang_view"
        )
    with col2:
        framework = st.text_input(
            "Framework/Library (if applicable)",
            key="code_framework_view",
            placeholder="e.g., PyTorch, TensorFlow, Scikit-learn, R"
        )

    if st.button("Generate Code ✨", key="code_gen_button_view"):
        if not code_request:
            st.warning("Please describe the code you want to generate.")
        elif code_chain and retriever:
            with st.spinner("Searching context and generating code..."):
                try:
                    # 1. Retrieve context relevant to the request
                    st.write("Finding relevant context...")
                    relevant_docs = retriever.invoke(code_request)
                    if not relevant_docs:
                         st.warning("Could not find specific context in the paper for your request. Proceeding with general knowledge (results may be less accurate).")
                         context_text = "No specific context found in the paper for this request."
                    else:
                         context_text = "\n\n---\n\n".join([f"Source Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
                         # Optionally show retrieved context
                         # with st.expander("Retrieved Context for Code Generation"):
                         #     st.text(context_text)

                    # 2. Invoke the code chain
                    st.write(f"Generating {language} code...")
                    generated_code = code_chain.invoke({
                        "language": language,
                        "framework": framework if framework else "Not specified",
                        "context": context_text,
                        "description": code_request,
                        "user_background": user_background # Pass user background to agent prompt
                    })

                    # Display code with language formatting
                    st.code(generated_code, language=language.lower() if language != "Other" else "plaintext")
                    st.success("Code generated! (Review carefully before use)")
                    st.caption("⚠️ AI-generated code may contain errors or inaccuracies. Always test thoroughly.")

                except Exception as e:
                    st.error(f"Error generating code: {e}")
                    logger.error(f"Code generation error: {e}")

        elif not retriever:
            st.error("Retriever not available. Please process a paper first.")
        else: # code_chain is None
            st.error("Code generation chain not available.")
    else:
         st.info("Describe the desired code, select language/framework, and click generate.")

