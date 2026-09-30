
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from schema import PromptRequest, PromptResponse
from laya_engine import LayaEngine

# Lifespan: Loads the model once into RAM when the server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = LayaEngine(device="cuda") # switch to "cuda" if using GPU
    yield

app = FastAPI(title="CircuitBreaker Gateway", lifespan=lifespan)

@app.post("/inspect", response_model=PromptResponse)
async def inspect_prompt(payload: PromptRequest):
    engine: LayaEngine = app.state.engine

    # Push CPU/GPU computation to a worker thread so the event loop stays unblocked
    result = await asyncio.to_thread(engine.evaluate, payload.prompt)
    
    answers = result["answers"]
    latency = result["latency_ms"]

    # Extract values
    p_injection = answers["is_prompt_injection"].get("noul", 0.0)
    intent = answers["intent_category"].get("choice", "chat")
    intent_conf = answers["intent_category"].get("confidence", 0.0)
    complexity = answers["complexity_score"].get("score", 0.0)

    # Policy 1: Security Drop
    if p_injection >= 0.85 or (intent == "malicious" and intent_conf >= 0.85):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Blocked by CircuitBreaker security policy",
                "injection_confidence": round(p_injection, 3),
                "latency_ms": round(latency, 2)
            }
        )

    # Policy 2: Complexity Routing
    if complexity < 1.0:
        action = "ROUTE_CACHE_OR_SLM"
    else:
        action = "ROUTE_HEAVY_LLM"

    return PromptResponse(
        is_injection=(p_injection >= 0.85),
        injection_confidence=round(p_injection, 3),
        intent=intent,
        intent_confidence=round(intent_conf, 3),
        complexity_score=round(complexity, 2),
        action=action,
        latency_ms=round(latency, 2)
    )