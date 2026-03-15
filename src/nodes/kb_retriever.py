"""Knowledge base retrieval node."""

from src.graph.state import AgentState
from src.services.faiss_store import FAISSStore


def retrieve_knowledge(state: AgentState) -> AgentState:
    """
    Retrieve relevant documents from FAISS knowledge base using semantic search.

    Searches FAISS index for semantically similar documents to the email content.

    Args:
        state: Current agent state with email body and intent

    Returns:
        Updated state with kb_results (list of relevant documents)
    """
    try:
        faiss_store = FAISSStore(index_path="data/faiss_index")

        # Search query combines subject and body for better relevance
        search_query = f"{state['subject']}. {state['body']}"

        # Retrieve top-5 relevant documents from FAISS
        results = faiss_store.search(query=search_query, top_k=5)

        state["kb_results"] = results

        if results:
            print(f"[INFO] Found {len(results)} relevant documents for intent: {state.get('intent')}")

    except Exception as e:
        print(f"[ERROR] KB retrieval failed: {e}")
        state["kb_results"] = []

    return state
