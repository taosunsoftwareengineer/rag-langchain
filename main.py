from initialize_llm import initialize_llm
from setup_vector_store import setup_vector_store
from rag_app import State
from langgraph.graph import START, StateGraph

llm, vector_store, prompt = initialize_llm()
vector_store = setup_vector_store(vector_store)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def main():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    result = graph.invoke({"question": "What is an LLM agent?"})
    print(result["answer"]) 
    
if __name__ == "__main__":
    main()