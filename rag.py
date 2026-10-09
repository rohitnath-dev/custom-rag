import os
import re
import json
import requests
from pathlib import Path
from typing import List, TypedDict

import streamlit as st
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START, END


# -------------------- PATHS --------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"


# -------------------- LLM --------------------

class OpenRouterLLM:
    def __init__(self, model="openai/gpt-4o-mini"):
        self.api_key = st.secrets["OPENROUTER_API_KEY"]
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def invoke(self, prompt):
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            },
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]


# -------------------- FILE LOADING --------------------

def load_configuration(identifier: str):
    identifier = str(identifier).strip()

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier):
        raise ValueError("Invalid identifier")

    data_path = DATA_DIR / f"{identifier}.txt"
    prompt_path = PROMPTS_DIR / f"{identifier}.txt"

    if not data_path.is_file() or not prompt_path.is_file():
        raise FileNotFoundError("Required configuration files are unavailable")

    data = data_path.read_text(encoding="utf-8").strip()
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    if not data or not system_prompt:
        raise ValueError("Configuration files must contain content")

    return data, system_prompt


# -------------------- RAG ENGINE --------------------

@st.cache_resource(show_spinner=False)
def create_rag_app(identifier: str):

    text, system_prompt = load_configuration(identifier)

    llm = OpenRouterLLM()

    documents = [
        Document(page_content=text)
    ]

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
    ).split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    # -------------------- STATE --------------------

    class State(TypedDict):
        question: str
        need_retrieval: bool
        docs: List[Document]
        relevant_docs: List[Document]
        context: str
        answer: str

    class RetrievalDecision(BaseModel):
        should_retrieve: bool = Field(...)

    class RelevanceDecision(BaseModel):
        is_relevant: bool

    # -------------------- DECISION --------------------

    def decide_retrieval(state: State):

        prompt = f"""
Determine whether the question requires retrieving information
from the provided knowledge collection.

Return only JSON:
{{"should_retrieve": true}}
or
{{"should_retrieve": false}}

Question:
{state["question"]}
"""

        response = llm.invoke(prompt)

        try:
            match = re.search(
                r"\{.*\}",
                response,
                re.DOTALL,
            )

            if match:
                result = RetrievalDecision(
                    **json.loads(match.group())
                )

                return {
                    "need_retrieval": result.should_retrieve
                }

        except (ValueError, TypeError, json.JSONDecodeError):
            pass

        return {"need_retrieval": True}

    # -------------------- DIRECT GENERATION --------------------

    direct_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("human", "{question}"),
    ])

    def generate_direct(state: State):

        messages = direct_prompt.format_messages(
            question=state["question"]
        )

        prompt = "\n".join(
            message.content for message in messages
        )

        return {
            **state,
            "answer": llm.invoke(prompt),
        }

    # -------------------- RETRIEVAL --------------------

    def retrieve(state: State):

        return {
            **state,
            "docs": retriever.invoke(
                state["question"]
            ),
        }

    # -------------------- RELEVANCE --------------------

    def check_relevance(state: State):

        relevant_docs = []

        for document in state["docs"]:

            prompt = f"""
Determine whether the supplied content is relevant
to answering the question.

Return only JSON:
{{"is_relevant": true}}
or
{{"is_relevant": false}}

Question:
{state["question"]}

Content:
{document.page_content}
"""

            response = llm.invoke(prompt)

            try:
                match = re.search(
                    r"\{.*\}",
                    response,
                    re.DOTALL,
                )

                if match:
                    result = RelevanceDecision(
                        **json.loads(match.group())
                    )

                    if result.is_relevant:
                        relevant_docs.append(document)

            except (ValueError, TypeError, json.JSONDecodeError):
                continue

        return {
            **state,
            "relevant_docs": relevant_docs,
        }

    # -------------------- CONTEXT GENERATION --------------------

    context_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}",
        ),
    ])

    def generate_from_context(state: State):

        context = "\n\n---\n\n".join(
            document.page_content
            for document in state["relevant_docs"]
        ).strip()

        messages = context_prompt.format_messages(
            question=state["question"],
            context=context,
        )

        prompt = "\n".join(
            message.content for message in messages
        )

        return {
            **state,
            "answer": llm.invoke(prompt),
            "context": context,
        }

    # -------------------- ROUTING --------------------

    def route_after_decision(state: State):

        if state["need_retrieval"]:
            return "retrieve"

        return "generate_direct"

    # -------------------- GRAPH --------------------

    graph = StateGraph(State)

    graph.add_node(
        "decide_retrieval",
        decide_retrieval,
    )

    graph.add_node(
        "generate_direct",
        generate_direct,
    )

    graph.add_node(
        "retrieve",
        retrieve,
    )

    graph.add_node(
        "check_relevance",
        check_relevance,
    )

    graph.add_node(
        "generate_from_context",
        generate_from_context,
    )

    graph.add_edge(
        START,
        "decide_retrieval",
    )

    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_decision,
        {
            "retrieve": "retrieve",
            "generate_direct": "generate_direct",
        },
    )

    graph.add_edge(
        "generate_direct",
        END,
    )

    graph.add_edge(
        "retrieve",
        "check_relevance",
    )

    graph.add_edge(
        "check_relevance",
        "generate_from_context",
    )

    graph.add_edge(
        "generate_from_context",
        END,
    )

    return graph.compile()