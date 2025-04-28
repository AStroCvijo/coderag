from rag.generate import *
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph
from handlers.model_handlers import get_embeddings_function

# ------------------------------------------------------------------------------------------------------
# VECTORSTORE QUERY
# ------------------------------------------------------------------------------------------------------

# Function for querying the vector store
def query_vector_store(query, persistent_directory, k, embedding_model):
    # Get embeddings function based on model type
    embeddings = get_embeddings_function(embedding_model)

    # Load the Chroma database
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings)

    # Perform similarity search
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k})
    # You can use expand query function here
    # Retrieve top-k documents
    docs = retriever.invoke(query)

    # Ensure unique documents before sorting
    unique_docs = {doc.metadata.get("source"): doc for doc in docs}.values()

    # Sort by relevance score in descending order
    sorted_docs = sorted(
        unique_docs, key=lambda x: x.metadata.get("score", 0.0), reverse=True
    )

    # Select only the top 10 documents
    top_docs = sorted_docs[:10]

    # Extract file names
    retrieved_files = [doc.metadata.get("source") for doc in top_docs]

    return retrieved_files

# ------------------------------------------------------------------------------------------------------
# VECTORSTORE QUERY WITH LLM SUMMARIES
# ------------------------------------------------------------------------------------------------------

# Function for generating LLM summaries of the retrieved code files
def query_vector_store_with_llm(
        query,
        persistent_directory,
        embedding_model,
        llm):
    # Get embeddings function based on model type
    embeddings = get_embeddings_function(embedding_model)

    # Load the Chroma database
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings)

    # Use the 'similarity' search type
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 100})

    (
        retrieval_grader,
        rag_chain,
        hallucination_grader,
        answer_grader,
        question_rewriter,
    ) = init_agents(llm)

    # Create a new workflow instance
    workflow = StateGraph(GraphState)

    # Add nodes to the workflow
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node(
        "grade_documents",
        lambda state: grade_documents(
            state,
            retrieval_grader))
    workflow.add_node(
        "generate",
        lambda state: generate(
            state,
            rag_chain,
            llm))
    workflow.add_node(
        "transform_query",
        lambda state: transform_query(
            state,
            question_rewriter))
    workflow.add_node(
        "out_of_scope_response",
        lambda state: out_of_scope_response(
            state,
            llm))

    # Set the entry point and define the edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "out_of_scope": "out_of_scope_response",
            "generate": "generate",
        },
    )
    workflow.add_edge("out_of_scope_response", END)
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        lambda state: grade_generation_v_documents_and_question(
            state, hallucination_grader, answer_grader
        ),
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile the workflow
    app = workflow.compile()

    # Genrate the answer
    inputs = {"question": query}
    for output in app.stream(inputs):
        for key, value in output.items():
            final_output = value

    return final_output["generation"]