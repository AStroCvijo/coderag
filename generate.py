import os
import json
import getpass
import tiktoken
from typing import List
from utils.const import *
from langchain import hub
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from utils.model_handlers import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Get the OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter API key for OpenAI: ")

def init_agents(model_name):
    # ------------------------------------------------------------------------------------------------------
    # RETRIEVAL GRADER
    # ------------------------------------------------------------------------------------------------------

    # Define a Pydantic model
    class GradeDocuments(BaseModel):
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # Initialize the LLM
    llm = get_llm(model_name)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Define the system prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    # ------------------------------------------------------------------------------------------------------
    # ANSWER GENERATOR
    # ------------------------------------------------------------------------------------------------------

    # Pull the RAG prompt from the hub
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------------------------------------------
    # HALLUCINATION GRADER
    # ------------------------------------------------------------------------------------------------------

    # Define a Pydantic model
    class GradeHallucinations(BaseModel):
        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Define the system prompt for
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader

    # ------------------------------------------------------------------------------------------------------
    # ANSWER GRADER
    # ------------------------------------------------------------------------------------------------------

    # Define a Pydantic model
    class GradeAnswer(BaseModel):
        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Define the system prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n LLM generation: {generation}",
            ),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader

    # ------------------------------------------------------------------------------------------------------
    # QUESTION RE-WRITER
    # ------------------------------------------------------------------------------------------------------

    # Define the system prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             system),
            ("human",
             "Here is the initial question: \n\n {question} \n Formulate an improved question.",
             ),
        ])

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    return (
        retrieval_grader,
        rag_chain,
        hallucination_grader,
        answer_grader,
        question_rewriter,
    )


# ------------------------------------------------------------------------------------------------------
# OUT OF SCOPE RESPONSE
# ------------------------------------------------------------------------------------------------------


# Function for handling out-of-scope questions
def out_of_scope_response(state, model_name):
    llm = get_llm(model_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """The user's question is outside the scope of the documents in the vectorstore. Politely inform them that their question is out of scope and cannot be answered based on the available information.
                      Dont engage in any type of conversation just inform them that their question is out of the scope of the vectorstore""",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"question": state["question"]})
    return {"generation": generation}


# ------------------------------------------------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------------------------------------------------


# Define the state structure for the graph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]


# ------------------------------------------------------------------------------------------------------
# GRAPH NODES
# ------------------------------------------------------------------------------------------------------


# Function for retrieving docs based on the question
def retrieve(state, retriever):
    question = state["question"]
    documents = retriever.invoke(question)

    # Replace the document's content with the actual content from the source
    # file
    for doc in documents:
        source_path = doc.metadata.get("absolute_path")
        if source_path:
            with open(source_path, "r") as file:
                actual_content = file.read()
            doc.page_content = actual_content

    return {"documents": documents, "question": question}


def count_tokens(text, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def trim_documents(documents, max_tokens, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    trimmed_documents = []
    current_token_count = 0

    for i, doc in enumerate(documents):
        # Serialize metadata to string
        metadata_str = json.dumps(doc.metadata)

        # Tokenize separately
        doc_tokens = encoding.encode(doc.page_content)
        metadata_tokens = encoding.encode(metadata_str)
        total_tokens = len(doc_tokens) + len(metadata_tokens)

        # Check if the document can fit within the remaining token limit
        if current_token_count + total_tokens <= max_tokens:
            trimmed_documents.append(doc)
            current_token_count += total_tokens
        else:
            continue

    return trimmed_documents


# Function for generating an answer using the RAG chain
def generate(state, rag_chain, llm):
    question = state["question"]
    documents = state["documents"]

    # Define the model's max context length
    max_context_length = MODEL_CONTEXT_LENGTHS.get(llm, 4096)

    # Calculate token count for the question
    question_tokens = count_tokens(question, llm)

    reserved_tokens = 500
    available_tokens = max_context_length - question_tokens - reserved_tokens

    # Trim documents to fit within the available tokens
    trimmed_documents = trim_documents(documents, available_tokens, llm)

    # Invoke the RAG chain with the trimmed documents
    generation = rag_chain.invoke(
        {"context": trimmed_documents, "question": question})

    return {
        "documents": trimmed_documents,
        "question": question,
        "generation": generation,
    }


# Function for grading the relevance of documents
def grade_documents(state, retrieval_grader):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score == "yes":
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": question}


# Function for transforming the query into a better version for retrieval
def transform_query(state, question_rewriter):
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": [], "question": better_question}


# ------------------------------------------------------------------------------------------------------
# GRAPH EDGES
# ------------------------------------------------------------------------------------------------------


# Function for deciding whether to generate an answer or handle an
# out-of-scope question
def decide_to_generate(state):
    if not state["documents"]:
        return "out_of_scope"
    return "generate"


# Function for grading the generation against the documents and the question
def grade_generation_v_documents_and_question(
    state, hallucination_grader, answer_grader
):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check if the generation is grounded in the documents
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if score.binary_score == "yes":
        # Check if the generation answers the question
        score = answer_grader.invoke(
            {"question": question, "generation": generation})
        if score.binary_score == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"
