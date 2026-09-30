from pydantic import BaseModel
from typing import Optional

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    is_injection: bool
    injection_confidence: float
    intent: str
    intent_confidence: float
    complexity_score: float
    action: str
    latency_ms: float
