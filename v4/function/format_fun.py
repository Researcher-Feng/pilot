from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional


# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    main_response: str

@dataclass
class ResponseFormat_detailed:
    """Response schema for the agent."""
    main_response: str
    detailed_response: str | None = None

@dataclass
class ResponseFormat_punny:
    """Response schema for the agent."""
    main_response: str
    weather_conditions: str | None = None
    punny_response: str | None = None

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


