from dataclasses import dataclass
from typing import Optional


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: Optional[str] = None
    user_role: Optional[str] = None




