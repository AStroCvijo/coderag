### Router

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from typing import List
from typing_extensions import TypedDict

# ------------------------------------------------------------------------------------------------------
# RETRIEVER
# ------------------------------------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

# Load the Chroma database
db = Chroma(persist_directory="db/chroma_viarotel-org/escrcpy_1200_200", embedding_function=embeddings)

# Use the 'similarity' search type
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# ------------------------------------------------------------------------------------------------------
# RETRIEVAL GRADER
# ------------------------------------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm_for_grading = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm_for_grading.with_structured_output(GradeDocuments)

system_grade = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grade),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# ------------------------------------------------------------------------------------------------------
# ANSWER GENERATOR
# ------------------------------------------------------------------------------------------------------

prompt = hub.pull("rlm/rag-prompt")
llm_for_generation = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
rag_chain = prompt | llm_for_generation | StrOutputParser()

# ------------------------------------------------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------------------------------------------------

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# ------------------------------------------------------------------------------------------------------
# GRAPH NODES
# ------------------------------------------------------------------------------------------------------

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}

def fallback(state):
    print("---FALLBACK ANSWER---")
    question = state["question"]
    # Provide a fallback context indicating no relevant docs were found.
    fallback_context = (
        "No relevant documents were found for this query. "
        "This question is outside the scope of the vector store retrieval."
    )
    generation = rag_chain.invoke({"context": fallback_context, "question": question})
    return {"documents": state["documents"], "question": question, "generation": generation}

# ------------------------------------------------------------------------------------------------------
# GRAPH EDGES
# ------------------------------------------------------------------------------------------------------

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    if not state["documents"]:
        print("---DECISION: NO RELEVANT DOCUMENTS, USE FALLBACK ANSWER---")
        return "fallback"
    print("---DECISION: GENERATE---")
    return "generate"

# ------------------------------------------------------------------------------------------------------
# WORKFLOW
# ------------------------------------------------------------------------------------------------------

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("fallback", fallback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"fallback": "fallback", "generate": "generate"},
)
# Both "generate" and "fallback" now directly produce the final answer (END)
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

# ------------------------------------------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------------------------------------------

app = workflow.compile()

# Run with a vectorstore-targeted question
inputs = {
    "question": "How is the wireless connection screen layout managed and updated across the application?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")
pprint(value["generation"])

# Run with a general question (no relevant documents; will trigger fallback)
inputs = {"question": "Hello"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")
pprint(value["generation"])
