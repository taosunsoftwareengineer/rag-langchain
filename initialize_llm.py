import getpass
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

def ensure_google_key() -> None:
    """Ensure the Google API key is set."""
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google: ")
    print("Google API key loaded successfully.")


def initialize_llm():
    """Initializes LangChain components with Gemini models."""
    # Ensure the API key is available
    ensure_google_key()

    # Initialize the Gemini chat model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

      # Initialize a local embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Define the vector store
    vector_store = InMemoryVectorStore(embeddings)

    # Define prompt template for question-answering
    prompt = hub.pull("rlm/rag-prompt")

    print("Gemini integration succeeded!")
    return llm, vector_store, prompt