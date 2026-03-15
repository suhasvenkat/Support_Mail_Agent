"""Utility functions for ID generation."""

import uuid
from datetime import datetime


def generate_id(prefix: str = "email") -> str:
    """
    Generate a unique ID with optional prefix.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_part = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_part}"
