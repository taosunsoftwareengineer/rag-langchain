
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START
from langchain_core.vectorstores import InMemoryVectorStore

def setup_vector_store(vector_store: InMemoryVectorStore) -> None:
    # Load and chun contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )

    # Use the loader to read and load the raw content from the source (e.g., a PDF file).
    docs = loader.load()

    # Initialize a text splitter to break down the loaded documents into smaller, manageable chunks.
    # chunk_size=1000 sets the maximum size of each chunk to 1000 characters.
    # chunk_overlap=200 sets an overlap of 200 characters between consecutive chunks to maintain context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Apply the text splitter to the loaded documents to create the chunks.
    all_splits = text_splitter.split_documents(docs)

    # Add all the newly created document splits (chunks) to the vector store.
    # This process embeds each chunk (converts it to a vector) and stores it for later retrieval.
    _ = vector_store.add_documents(documents=all_splits)
    
    print("Vector store is initialized")
    return vector_store