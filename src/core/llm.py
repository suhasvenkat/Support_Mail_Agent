from src.core.config import get_settings
from src.services.mock_llm import MockLLM

def get_llm(temperature: float = 0.7):
    settings = get_settings()
    
    if settings.mock_mode:
        return MockLLM(temperature=temperature)
    
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="mistral", temperature=temperature)
        print("[INFO] Using Ollama Mistral (local, free)")
        return llm
    except Exception as e:
        print(f"[WARN] Ollama failed: {e}")
    
    print("[INFO] Using MOCK LLM (no API calls)")
    return MockLLM(temperature=temperature)
