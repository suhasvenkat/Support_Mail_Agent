"""LLM initialization and factory functions."""

from src.core.config import get_settings

# Try real LLM, provide mock fallback
try:
    from langchain_openai import ChatOpenAI
    HAS_REAL_LLM = True
except ImportError:
    HAS_REAL_LLM = False

from src.services.mock_llm import MockLLM


def get_llm(temperature: float = 0.7):
    """
    Get a configured LLM instance (real or mock).

    Args:
        temperature: Controls randomness (0.0-1.0). Lower = more deterministic.

    Returns:
        ChatOpenAI or MockLLM instance
    """
    settings = get_settings()

    # Use mock mode if enabled or if no API key available
    if settings.mock_mode or not settings.openai_api_key:
        print("[INFO] Using MOCK LLM (no API calls)")
        return MockLLM(temperature=temperature)

    try:
        print("[INFO] Using REAL OpenAI LLM")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
    except Exception as e:
        print(f"[WARN] OpenAI LLM failed: {e}")
        print("[INFO] Falling back to MOCK LLM")
        return MockLLM(temperature=temperature)
