"""
SEED AI — Orchestration API (Production v3 — Auth-Enforced)
SSE streaming endpoints with request ID tracing, guaranteed response,
and strict user_id verification against the Firebase UID from the auth token.
"""
import uuid
import json
import base64
import traceback

from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from agents.orchestrator_agent import OrchestratorAgent
from middleware import require_auth
from services.firebase_service import is_available as firebase_available

router = APIRouter()
orchestrator = OrchestratorAgent()


class OrchestrateRequest(BaseModel):
    user_id: str
    text_query: Optional[str] = None
    image_base64: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    crop: Optional[str] = None
    execution_mode: Optional[str] = "auto"
    specific_agent: Optional[str] = None


@router.post("/orchestrate")
async def orchestrate_workflow(
    request: OrchestrateRequest,
    user: dict = Depends(require_auth),
):
    auth_uid = user.get("uid", "anonymous")
    effective_uid = auth_uid if auth_uid != "anonymous" else (request.user_id or "anonymous")
    if auth_uid != "anonymous" and request.user_id != auth_uid:
        return {"error": "Forbidden: user_id must match authenticated user"}

    context = request.model_dump()
    context["user_id"] = effective_uid
    context["request_id"] = str(uuid.uuid4())[:8]

    async def event_generator():
        try:
            async for event in orchestrator.process_stream(context):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_detail = {
                "type": "error",
                "message": f"Orchestrator error: {str(e)}",
                "status": "failed",
            }
            yield f"data: {json.dumps(error_detail)}\n\n"
            yield f"data: {json.dumps({'type': 'result', 'data': {'summary': 'The simulation encountered an unexpected error. Please retry with a more specific query.', 'recommended_actions': ['Retry the simulation', 'Try a different execution mode', 'Check API key configuration'], 'confidence_score': 0, 'confidence_label': 'Low', 'risk_level': 'Unknown', 'is_degraded': True}, 'execution_metadata': {'error': str(e), 'is_degraded': True, 'routed_mode': 'error'}})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/orchestrate/upload")
async def orchestrate_with_image(
    user: dict = Depends(require_auth),
    user_id: str = Form(...),
    text_query: str = Form(""),
    location: str = Form(""),
    budget: float = Form(0),
    crop: str = Form(""),
    execution_mode: str = Form("auto"),
    specific_agent: str = Form(""),
    image: UploadFile = File(None),
):
    auth_uid = user.get("uid", "anonymous")
    effective_uid = auth_uid if auth_uid != "anonymous" else (user_id or "anonymous")
    if auth_uid != "anonymous" and user_id != auth_uid:
        return {"error": "Forbidden: user_id must match authenticated user"}

    context = {
        "user_id": effective_uid,
        "text_query": text_query,
        "location": location,
        "budget": budget,
        "crop": crop,
        "execution_mode": execution_mode,
        "specific_agent": specific_agent,
        "request_id": str(uuid.uuid4())[:8],
    }

    if image and image.filename:
        try:
            image_bytes = await image.read()
            context["image_bytes"] = image_bytes
            context["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            context["image_upload_error"] = str(e)

    async def event_generator():
        try:
            async for event in orchestrator.process_stream(context):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_detail = {
                "type": "error",
                "message": f"Orchestrator error: {str(e)}",
                "status": "failed",
            }
            yield f"data: {json.dumps(error_detail)}\n\n"
            yield f"data: {json.dumps({'type': 'result', 'data': {'summary': 'Simulation encountered an error during image processing. Try again without an image.', 'recommended_actions': ['Retry without image upload', 'Use a smaller image file', 'Check API configuration'], 'confidence_score': 0, 'confidence_label': 'Low', 'risk_level': 'Unknown', 'is_degraded': True}, 'execution_metadata': {'error': str(e), 'is_degraded': True, 'routed_mode': 'error'}})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
