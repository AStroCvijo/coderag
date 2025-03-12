from langchain import hub
from typing import Literal, List
from langchain_chroma import Chroma
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------------------------------------------
# RETRIEVAL GRADER
# ------------------------------------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# ------------------------------------------------------------------------------------------------------
# ANSWER GENERATOR
# ------------------------------------------------------------------------------------------------------

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

# ------------------------------------------------------------------------------------------------------
# HALLUCINATION GRADER
# ------------------------------------------------------------------------------------------------------

class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

# ------------------------------------------------------------------------------------------------------
# ANSWER GRADER
# ------------------------------------------------------------------------------------------------------

class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

# ------------------------------------------------------------------------------------------------------
# QUESTION RE-WRITER
# ------------------------------------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

# ------------------------------------------------------------------------------------------------------
# OUT OF SCOPE RESPONSE
# ------------------------------------------------------------------------------------------------------

def out_of_scope_response(state):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """The user's question is outside the scope of the documents in the vectorstore. Politely inform them that their question is out of scope and cannot be answered based on the available information.
                      Dont engage in any type of conversation just inform them that their question is out of the scope of the vectorstore"""),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"question": state["question"]})
    return {"generation": generation}

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

def retrieve(state, retriever):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": [], "question": better_question}

# ------------------------------------------------------------------------------------------------------
# GRAPH EDGES
# ------------------------------------------------------------------------------------------------------

def decide_to_generate(state):
    if not state["documents"]:
        return "out_of_scope"
    return "generate"

def grade_generation_v_documents_and_question(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if score.binary_score == "yes":
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"
