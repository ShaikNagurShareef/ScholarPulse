# Filename: core/agents.py

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# For summarization if needed for longer docs:
# from langchain.chains.summarize import load_summarize_chain

from core.llm_setup import get_llm # Import our LLM setup function

# LangGraph imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

import logging
from utils import get_logger

# Setup logging using custom logger
logger = get_logger(__name__)

# --- Question-Answering LangGraph Agent ---

def create_agentic_rag_chain(retriever, user_background="knowledgeable user"):
    logger.info(f"Creating agentic RAG chain for user background: {user_background}")

    # Define retrieval tool with improved context
    def retrieve(query: str) -> str:
        """Retrieve detailed information from the paper relevant to the query."""
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            return "No relevant context found in the paper."
        # Increase context detail by joining more content
        serialized = "\n\n".join(
            f"Chunk {i+1} (Source: Page {doc.metadata.get('page', 'unknown')}):\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs[:5])  # Limit to 5 chunks for token safety
        )
        return serialized

    tools = [retrieve]
    tool_node = ToolNode(tools)

    # Initialize LLM with tools
    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools)
    except ValueError as e:
        logger.error(f"Error getting LLM for agentic RAG chain: {e}")
        raise

    # Define the graph
    graph_builder = StateGraph(MessagesState)

    # Node 1: Process the query
    def query_or_respond(state: MessagesState):
        logger.info(f"Query or respond state: {[msg.type for msg in state['messages']]}")
        system_prompt = (
            f"You are ScholarPulse, an AI Research Assistant designed to help users understand a research paper. "
            f"Answer the user’s query fully and concisely (2-5 sentences). "
            f"If the query is about the paper, use the 'retrieve' tool to fetch relevant context. "
            f"If it’s a general or off-topic question, respond conversationally without retrieval. "
            f"Tailor your response to a '{user_background}' and ensure it’s complete."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        logger.info(f"LLM response: type={type(response)}, content={response.content}, tool_calls={getattr(response, 'tool_calls', 'None')}")
        return {"messages": [response]}

    # Node 2: Generate the final response
    def generate_response(state: MessagesState):
        tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
        context = "\n\n".join(msg.content for msg in tool_messages) if tool_messages else "No context retrieved."
        logger.info(f"Retrieved context: {context[:200]}...")

        system_message = SystemMessage(content=(
            f"You are ScholarPulse, an AI Research Assistant for answering questions about a research paper. "
            f"Use the retrieved context below to answer the query fully and concisely (2-5 sentences). "
            f"If no context is retrieved and the query is paper-related, say 'I couldn’t find specific details in the paper.' "
            f"For non-paper queries, respond conversationally. "
            f"Tailor answers to a '{user_background}' and cut straight to the point, no yapping.\n\n"
            f"Retrieved Context:\n{context}"
        ))

        conversation = [msg for msg in state["messages"] if msg.type in ("human", "system", "ai") and not getattr(msg, 'tool_calls', None)]
        prompt = [system_message] + conversation
        logger.info(f"Prompt length: {len(prompt)} messages")

        response = llm.invoke(prompt)
        if not response.content.strip():
            response.content = "I’m sorry, I couldn’t generate a complete response. Please try rephrasing your question!"
        return {"messages": [response]}

    # Add nodes
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("generate", generate_response)

    # Define flow
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {"tools": "tools", END: "generate"}
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # Add memory
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    logger.info("Agentic RAG chain created successfully.")
    return graph

# --- Question-Answering (RAG) Agent ---

def create_rag_chain(retriever, user_background="knowledgeable user"):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain for Q&A.

    Args:
        retriever: An initialized LangChain retriever instance (from core.retrieval).
        user_background (str): A description of the user's background to tailor the response.

    Returns:
        Runnable: A LangChain runnable chain ready to process queries.
                 The chain expects a dictionary with "input" (the question)
                 and optionally uses "user_background" if included in the prompt.
    """
    logger.info(f"Creating RAG chain for user background: {user_background}")

    # 1. Define the Prompt Template
    # This template instructs the LLM how to use the retrieved context.
    # It includes placeholders for 'context' (retrieved documents) and 'input' (user question).
    # We also add the user_background to guide the tone and complexity.

    template = """You are an assistant for answering questions about a research paper.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Keep the answer concise and directly relevant to the question.
    Tailor your explanation to someone with the background of a '{user_background}' and cut straight to the point, no yapping..

    Context:
    {context}

    Question: {input}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 2. Get the LLM instance
    try:
        llm = get_llm() #get_codellm
    except ValueError as e:
        logger.error(f"Error getting LLM for RAG chain: {e}")
        raise # Re-raise the error to be handled upstream

    # 3. Create the "stuff documents" chain
    # This chain takes retrieved documents, formats them into the prompt's 'context' variable,
    # and sends the final prompt to the LLM.
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser() # Parses the LLM output into a string
    )

    # 4. Create the main Retrieval Chain
    # This chain orchestrates the process:
    # - Takes the user 'input' question.
    # - Passes it to the 'retriever' to get relevant documents.
    # - Passes the original 'input' and the retrieved 'context' to the 'combine_docs_chain'.
    # We add RunnablePassthrough to ensure the original 'input' and 'user_background'
    # are available to the final combine_docs_chain.
    retrieval_chain = create_retrieval_chain(
        # Pass 'input' to the retriever AND through to the combine_docs_chain
        retriever=retriever,
        # The document chain defined above
        combine_docs_chain=combine_docs_chain
    )

    # Inject the user_background into the chain's execution context if needed explicitly by prompt
    # (The current prompt template already includes it directly)
    # If the prompt didn't have user_background, you might do something like:
    # setup_and_retrieval = RunnableParallel(
    #     {"context": retriever, "input": RunnablePassthrough(), "user_background": lambda x: user_background}
    # )
    # retrieval_chain = setup_and_retrieval | combine_docs_chain

    logger.info("RAG chain created successfully.")
    return retrieval_chain


# --- Summarization Agent ---

def create_summarization_chain(user_background="knowledgeable user", chain_type="stuff"):
    """
    Creates a chain for summarizing text (e.g., the whole paper or key sections).

    Args:
        user_background (str): Description of the user's background for tailoring.
        chain_type (str): Type of chain ('stuff', 'map_reduce', 'refine').
                          'stuff' is simplest if text fits context window.

    Returns:
        Runnable: A LangChain runnable chain ready to process text for summarization.
                  Expects a dictionary with "input_documents".
    """
    logger.info(f"Creating Summarization chain (type: {chain_type}) for user background: {user_background}")

    # 1. Get the LLM instance
    try:
        llm = get_llm()
    except ValueError as e:
        logger.error(f"Error getting LLM for Summarization chain: {e}")
        raise

    # 2. Define the Prompt Template (specific to summarization)
    # Using PromptTemplate for potentially simpler single-input scenarios
    prompt_template = """Write a detailed yet accessible summary of the following text, tailored to someone with the background of a '{user_background}'. 
    Focus on clearly explaining the key findings, methodology, results, and conclusions of the paper. 
    Include all important technical details—such as theoretical foundations, experimental design, 
    or supporting evidence—and break them down into simple, easy-to-understand language while keeping the explanation convincing and comprehensive. 
    Additionally, summarize the literature provided in the paper, highlight the gaps or limitations it discusses, and explain how the paper addresses those gaps. 
    Ensure the summary is thorough, covering the results in detail and leaving no major aspect of the paper unexplained, so the reader fully grasps its content, significance, and contributions.

    Text:

    "{text}"

    Detailed Summary:"""

    prompt = PromptTemplate.from_template(prompt_template)

    # 3. Create the appropriate chain based on type
    if chain_type == "stuff":
        # Simple chain that "stuffs" all text into the context window.
        # Good for shorter texts that fit within the LLM's limit.
        summarize_chain = prompt | llm | StrOutputParser()

        # To use with documents directly (if input is List[Document]):
        # You'd typically use load_summarize_chain for this
        # from langchain.chains.summarize import load_summarize_chain
        # summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    elif chain_type == "map_reduce":
        # Handles longer documents: Summarizes chunks individually (map), then combines summaries (reduce).
        # Requires defining map and combine prompts.
        logger.warning("Note: 'map_reduce' summarization chain setup is more complex and not fully implemented here.")
        logger.warning("Falling back to 'stuff' chain logic for this example.")
        # Example placeholder - requires map_prompt and combine_prompt definition
        # summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summarize_chain = prompt | llm | StrOutputParser() # Fallback for now

    else:
        raise ValueError(f"Unsupported summarization chain type: {chain_type}")

    logger.info("Summarization chain created successfully.")

    # How this chain might be invoked (adapt based on actual implementation):
    # If using the simple LCEL chain:
    # result = summarize_chain.invoke({"text": your_full_text, "user_background": user_background})

    # If using load_summarize_chain:
    # result = summarize_chain.invoke(input_documents=list_of_document_objects, user_background=user_background)
    # Need to ensure user_background is passed correctly if the chain supports it or modify the prompts

    # For now, returning the simpler LCEL chain, assuming text fits context.
    # The caller will need to handle providing the text and background correctly.
    # We wrap it slightly to match expected input format if needed later
    # This wrapper assumes the chain expects a 'text' variable.
    final_chain = RunnablePassthrough.assign(
         output = (lambda x: summarize_chain.invoke({"text": x["input_text"], "user_background": user_background}))
    )
    # This ^^ might be overly complex. The simple `prompt | llm | StrOutputParser`
    # is often sufficient if invoked correctly. Let's return that directly.
    return prompt | llm | StrOutputParser()


# --- Code Generation Agent (Placeholder) ---

def create_code_generation_chain(user_background="programmer"):
    """
    (Placeholder) Creates a chain for generating code based on paper context.

    Args:
        user_background (str): User's programming background (helps tailor comments/style).

    Returns:
        Runnable: A LangChain runnable chain (structure TBD).
    """
    logger.info("Creating Code Generation chain...")
    # 1. Get LLM
    try:
        llm = get_llm()
    except ValueError as e:
        logger.error(f"Error getting LLM for Code Gen chain: {e}")
        raise

    # 2. Define Prompt
    # Needs placeholders for language, framework, description, and relevant paper context
    template = """You are an AI assistant helping implement algorithms from a research paper.
    Generate code based on the provided description and context from the paper.
    The target language is '{language}' and the preferred framework/library is '{framework}'.
    Ensure the code is well-commented, explaining the steps, considering the user is a '{user_background}'.

    Relevant Paper Context:
    {context}

    Code Description / Request:
    {description}

    Generated {language} Code:
    ```python
    # Start your code here

    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Create Chain
    # This is likely a direct LLM call with the formatted prompt.
    # Retrieval of the 'context' might happen *before* calling this chain based on the 'description'.
    code_chain = prompt | llm | StrOutputParser()

    logger.info("Code Generation chain created.")
    return code_chain

if __name__ == '__main__':
    logger.info("--- Testing Agent Creation ---")

    # Mock Retriever class for testing
    class MockRetriever:
        def invoke(self, input_str):
            logger.info(f"MockRetriever invoked with: '{input_str}'")
            # Return mock LangChain Document objects
            return [
                Document(page_content="Context chunk 1 about self-attention."),
                Document(page_content="Context chunk 2 relevant to the question."),
            ]
        def get_relevant_documents(self, query): # Older interface, invoke is preferred
            return self.invoke(query)

    mock_retriever = MockRetriever()
    test_user_background = "Master's Student in Computer Science"

    # Test RAG Chain creation
    logger.info("Testing RAG Chain...")
    try:
        rag_chain = create_rag_chain(mock_retriever, test_user_background)
        logger.info(f"RAG Chain Type: {type(rag_chain)}")
        # Test invocation (won't actually call LLM unless you have keys setup)
        # response = rag_chain.invoke({"input": "What is self-attention?"})
        # logger.info(f"Mock RAG Response structure (if invoked): {response}") # Response structure depends on create_retrieval_chain version
    except Exception as e:
        logger.error(f"Error creating/testing RAG chain: {e}")

    # Test Summarization Chain creation
    logger.info("Testing Summarization Chain...")
    try:
        summarize_chain = create_summarization_chain(test_user_background)
        logger.info(f"Summarization Chain Type: {type(summarize_chain)}")
        # Test invocation
        # summary = summarize_chain.invoke({"input_text": "This is a long text about a paper...", "user_background": test_user_background}) # Adapting input dict
        # logger.info(f"Mock Summary (if invoked): {summary}")
    except Exception as e:
        logger.error(f"Error creating/testing Summarization chain: {e}")