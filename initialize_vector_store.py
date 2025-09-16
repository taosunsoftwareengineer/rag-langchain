
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore

def initialize_vector_store(vector_store: InMemoryVectorStore) -> None:
    # Load and chun contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_splits = text_splitter.split_documents(docs)

    _ = vector_store.add_documents(documents=all_splits)
    
    print("Vector store is initialized")

    prompt = hub.pull("rlm/rag-prompt")
    return vector_store


