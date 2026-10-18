"""Errors that API and LangGraph can import without initializing Gemini."""


class GeminiUnavailable(RuntimeError):
    """Gemini is unavailable or returned an unusable response."""
