"""
SEED AI — API Routes (Production)
Enhanced health, diagnostics, and data endpoints.
"""
import asyncio
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Header, Depends, File, Form, UploadFile
from typing import Optional
import sys
import os

from .orchestration import router as orchestrate_router

logger = logging.getLogger("Routes")
from utils.dataset_manager import DatasetManager
from services.firebase_service import (
    health_check as firebase_health,
    verify_token_cached,
    is_available as firebase_available,
)
from services import gemini_service
from services.hybrid_vision_service import HybridVisionService
from services.weather_service import WeatherService
from services.market_service import MarketService
from agents.memory_agent import MemoryAgent
from evaluation.benchmark import AgentEvaluator
from middleware import require_auth, optional_auth

router = APIRouter()
router.include_router(orchestrate_router, tags=["Orchestration"])

_dm = DatasetManager()
_memory = MemoryAgent()
_evaluator = AgentEvaluator.get_shared()


@router.get("/health")
def health_check():
    """
    Lightweight health endpoint — NO Gemini API ping.
    Safe to poll every 10s without consuming API quota.
    """
    firebase_status = firebase_health()
    gemini_ok = gemini_service.is_available()
    firestore_ok = firebase_status.get("firestore", False)
    overall = "healthy" if gemini_ok else "degraded"
    if not gemini_ok and not firestore_ok:
        overall = "unhealthy"

    return {
        "status": overall,
        "service": "SEED AI",
        "version": "2.0.0",
        "agents": [
            "orchestrator", "vision", "weather", "market",
            "budget", "crop_knowledge", "government_schemes",
            "task_planning", "translation", "memory",
            "crop_prediction", "disease_prediction", "waste_to_wealth",
            "soil_nutrient", "entomologist", "irrigation",
        ],
        "knowledge_bases": _dm.list_categories(),
        "gemini": {
            "available": gemini_ok,
            "model": gemini_service.DEFAULT_MODEL,
            "error": gemini_service.get_init_error(),
        },
        "firebase": firebase_status,
        "features": {
            "function_calling": gemini_ok,
            "parallel_execution": True,
            "streaming": True,
            "vision": gemini_ok,
            "memory": firestore_ok,
            "storage": True,
        },
    }


@router.get("/health/deep")
def health_check_deep():
    """
    Deep health check — includes a live Gemini API ping.
    Use sparingly (consumes 1 API call from rate limit).
    """
    firebase_status = firebase_health()
    gemini_status = gemini_service.health_check()

    gemini_ok = gemini_status.get("available", False)
    firestore_ok = firebase_status.get("firestore", False)
    overall = "healthy" if gemini_ok else "degraded"
    if not gemini_ok and not firestore_ok:
        overall = "unhealthy"

    return {
        "status": overall,
        "gemini": gemini_status,
        "firebase": firebase_status,
    }


@router.get("/diagnostics")
def diagnostics():
    """
    Configuration diagnostics — shows what's configured (not values).
    Safe to expose: no secrets leaked.
    """
    env_checks = {
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "GOOGLE_APPLICATION_CREDENTIALS": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        "OPENWEATHER_API_KEY": bool(os.getenv("OPENWEATHER_API_KEY")),
        "DATA_GOV_IN_API_KEY": bool(os.getenv("DATA_GOV_IN_API_KEY")),
        "SUPABASE_URL": bool(os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")),
        "DATASET_PATH": bool(os.getenv("DATASET_PATH")),
    }

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key.startswith("AIzaSyBk_placeholder"):
        env_checks["GEMINI_API_KEY_VALID"] = False
    else:
        env_checks["GEMINI_API_KEY_VALID"] = bool(gemini_key)

    return {
        "environment": env_checks,
        "python_version": sys.version,
        "gemini_model": gemini_service.DEFAULT_MODEL,
        "gemini_available": gemini_service.is_available(),
        "gemini_init_error": gemini_service.get_init_error(),
        "firebase_available": firebase_available(),
        "knowledge_bases": _dm.list_categories(),
        "knowledge_base_entries": {
            cat: len(_dm.get_all(cat)) if isinstance(_dm.get_all(cat), list) else "dict"
            for cat in _dm.list_categories()
        },
    }


@router.get("/knowledge/{category}")
def get_knowledge(category: str):
    data = _dm.get_all(category)
    if not data:
        return {"error": f"Category '{category}' not found", "available": _dm.list_categories()}
    return {"category": category, "data": data}


@router.get("/history/{user_id}")
async def get_history(
    user_id: str,
    authorization: Optional[str] = Header(None),
    page: int = 1,
    limit: int = 20,
    sort: str = "newest",
    search: Optional[str] = None,
    confidence: Optional[str] = None,
):
    """
    Returns conversation history for the authenticated user from Firestore.
    Security: The authenticated user's UID must match the requested user_id.
    Falls back to anonymous if Firebase auth is not configured.
    Supports pagination, sorting, search, and confidence filtering.
    """
    auth_uid = None

    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        claims = verify_token_cached(token) if firebase_available() else None
        if claims:
            auth_uid = claims.get("uid", "")
            if auth_uid and auth_uid != user_id:
                return {"error": "Forbidden: you can only access your own history", "history": [], "total": 0, "page": page, "limit": limit}
        elif firebase_available():
            return {"error": "Invalid or expired authentication token", "history": [], "total": 0, "page": page, "limit": limit}

    effective_user_id = auth_uid or user_id

    result = _memory.process({
        "user_id": effective_user_id,
        "memory_action": "get_history",
        "page": page,
        "limit": limit,
        "sort": sort,
        "search": search,
        "confidence": confidence,
    })
    return result


@router.get("/history/me")
async def get_my_history(
    authorization: Optional[str] = Header(None),
    page: int = 1,
    limit: int = 20,
    sort: str = "newest",
    search: Optional[str] = None,
    confidence: Optional[str] = None,
):
    """
    Returns conversation history for the currently authenticated user.
    Uses the Firebase token to determine the user ID automatically.
    """
    if firebase_available() and authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        claims = verify_token_cached(token)
        if claims:
            uid = claims.get("uid", "")
            result = _memory.process({
                "user_id": uid,
                "memory_action": "get_history",
                "page": page,
                "limit": limit,
                "sort": sort,
                "search": search,
                "confidence": confidence,
            })
            return result
        return {"error": "Invalid or expired authentication token", "history": [], "total": 0, "page": page, "limit": limit}

    return {"error": "Authentication required", "history": [], "total": 0, "page": page, "limit": limit}


@router.get("/evaluation/traces")
async def get_evaluation_traces(
    user: dict = Depends(require_auth),
    user_id: Optional[str] = None,
):
    """
    Returns evaluation traces scoped to the authenticated user.
    Reads from Firestore first (survives restarts),
    falls back to in-memory evaluator if Firestore is unavailable.
    Enforces strict UID ownership — the caller can only see their own data.
    """
    auth_uid = user.get("uid", "anonymous")

    if user_id and user_id != auth_uid and user.get("firebase_available") is not False:
        return {"error": "Forbidden: you can only access your own traces", "traces": [], "count": 0}

    effective_user_id = user_id if user_id == auth_uid else auth_uid

    if firebase_available() and _memory.db:
        try:
            result = await asyncio.to_thread(
                _memory.process,
                {"memory_action": "get_traces", "user_id": effective_user_id, "limit": 100},
            )
            if result.get("traces"):
                return {"traces": result["traces"], "count": result["count"], "source": "firestore"}
        except Exception as e:
            logger.warning(f"Firestore trace read failed, falling back to memory: {e}")

    traces = _evaluator.get_user_traces(effective_user_id)
    trace_list = [t["trace"] for t in traces]
    return {
        "traces": trace_list,
        "count": len(trace_list),
        "source": "memory",
    }


@router.get("/health/connectivity")
async def connectivity_check():
    """Deep connectivity check — tests Firestore write/read and EventBus health."""
    import random

    checks = {}

    # Test EventBus
    from services.bus import is_initialized as bus_initialized, get_bus
    if bus_initialized():
        bus = get_bus()
        checks["event_bus"] = "running"
        checks["event_bus_metrics"] = bus.metrics
    else:
        checks["event_bus"] = "not_initialized"

    # Test Firestore write
    if firebase_available():
        test_id = f"health_test_{int(time.time())}"
        try:
            import asyncio
            result = await asyncio.to_thread(
                _memory.process,
                {"memory_action": "batch_persist", "user_id": test_id},
            )
            checks["firestore_write"] = result.get("status") in ("batch_persisted", "dlq_queued")
        except Exception as e:
            checks["firestore_write"] = False
            checks["firestore_write_error"] = str(e)[:100]
    else:
        checks["firestore_write"] = False
        checks["firestore_write_error"] = "Firebase not available"

    checks["gemini_available"] = gemini_service.is_available()
    checks["in_memory_traces"] = len(_evaluator.trace_data)
    checks["all_ok"] = (
        checks.get("event_bus") == "running"
        and checks.get("firestore_write") is True
        and checks["gemini_available"]
    )

    return checks


@router.get("/activity/stream")
async def stream_activity(
    authorization: Optional[str] = Header(None),
):
    """
    SSE endpoint to push real-time execution events for the authenticated user only.
    Subscribes to EventBus for push notifications, falls back to polling.
    Enforces strict UID isolation — only the caller's events are pushed.
    """
    import asyncio
    import json
    from fastapi.responses import StreamingResponse

    # Extract auth UID from token
    auth_uid = "anonymous"
    if firebase_available() and authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        claims = verify_token_cached(token)
        if claims:
            auth_uid = claims.get("uid", "anonymous")
    
    async def event_generator():
        user_traces = _evaluator._get_user_traces(auth_uid)
        last_index = len(user_traces)
        
        # Yield initial state
        yield f"data: {json.dumps({'type': 'init', 'count': last_index, 'user_id': auth_uid})}\n\n"

        # Try to subscribe to EventBus for push notifications
        bus_event_queue: asyncio.Queue = asyncio.Queue()
        bus_subscribed = False

        from services.bus import is_initialized as bus_initialized, get_bus
        if bus_initialized():
            try:
                bus = get_bus()

                async def on_persist_event(event):
                    await bus_event_queue.put(event)

                bus.subscribe("persist", on_persist_event)
                bus_subscribed = True
            except Exception:
                pass

        try:
            while True:
                # Check bus queue first (non-blocking)
                new_event = None
                if bus_subscribed:
                    try:
                        new_event = bus_event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                if new_event:
                    # A new persist event means new trace data is available
                    user_traces = _evaluator._get_user_traces(auth_uid)
                    current_len = len(user_traces)
                    if current_len > last_index:
                        new_traces = user_traces[last_index:current_len]
                        last_index = current_len
                        if new_traces:
                            yield f"data: {json.dumps({'type': 'new_traces', 'traces': new_traces, 'user_id': auth_uid})}\n\n"
                else:
                    # Fallback polling every 2s
                    user_traces = _evaluator._get_user_traces(auth_uid)
                    current_len = len(user_traces)
                    if current_len > last_index:
                        new_traces = user_traces[last_index:current_len]
                        last_index = current_len
                        if new_traces:
                            yield f"data: {json.dumps({'type': 'new_traces', 'traces': new_traces, 'user_id': auth_uid})}\n\n"

                await asyncio.sleep(1)
        finally:
            if bus_subscribed:
                try:
                    bus.unsubscribe("persist", on_persist_event)
                except Exception:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/trends")
async def get_trends(
    user: dict = Depends(require_auth),
    days: int = 7,
):
    """
    Returns aggregated execution trends over the last N days for the authenticated user.
    Used by the dashboard for 7-day trend charts.
    Enforces strict UID isolation.
    """
    from datetime import timedelta
    auth_uid = user.get("uid", "anonymous")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    if not firebase_available() or not _memory.db:
        user_traces = _evaluator.get_user_traces(auth_uid)
        all_traces = [t["trace"] for t in user_traces]
        return {
            "daily_executions": [],
            "daily_avg_latency": [],
            "daily_success_rate": [],
            "source": "memory",
            "total_traces": len(all_traces),
        }

    try:
        import asyncio

        traces_result = await asyncio.to_thread(
            _memory.process,
            {"memory_action": "get_traces", "user_id": auth_uid, "limit": 1000},
        )
        all_traces = traces_result.get("traces", [])

        # Filter to last N days
        recent = [
            t for t in all_traces
            if t.get("timestamp", "") >= cutoff_str
        ]

        # Group by day
        from collections import defaultdict
        daily = defaultdict(lambda: {"count": 0, "total_latency": 0.0, "successes": 0, "total_agents": 0})

        for t in recent:
            day = t.get("timestamp", "")[:10]
            daily[day]["count"] += 1
            daily[day]["total_latency"] += t.get("total_latency_sec", 0) or 0

            agents = t.get("agents_invoked", []) or t.get("agents", [])
            for a in agents:
                daily[day]["total_agents"] += 1
                agent_ok = a.get("success") if a.get("success") is not None else (a.get("status") == "completed")
                if agent_ok:
                    daily[day]["successes"] += 1

        sorted_days = sorted(daily.keys())

        return {
            "daily_executions": [
                {"date": d, "count": daily[d]["count"]} for d in sorted_days
            ],
            "daily_avg_latency": [
                {
                    "date": d,
                    "avg_latency_sec": round(
                        daily[d]["total_latency"] / daily[d]["count"], 2
                    ) if daily[d]["count"] else 0,
                }
                for d in sorted_days
            ],
            "daily_success_rate": [
                {
                    "date": d,
                    "success_rate": round(
                        (daily[d]["successes"] / daily[d]["total_agents"]) * 100, 1
                    ) if daily[d]["total_agents"] else 0,
                }
                for d in sorted_days
            ],
            "source": "firestore",
        }
    except Exception as e:
        logger.error(f"Trends error: {e}")
        return {
            "daily_executions": [],
            "daily_avg_latency": [],
            "daily_success_rate": [],
            "error": str(e)[:100],
        }


@router.get("/anomalies")
async def detect_anomalies(
    user: dict = Depends(require_auth),
    lookback: int = 20,
):
    """
    Detect latency anomalies using a rolling 2-sigma threshold.
    Flags any execution with latency > 2 standard deviations from the
    rolling mean of the last N executions. Scoped to the authenticated user.
    """
    auth_uid = user.get("uid", "anonymous")
    user_traces_data = _evaluator.get_user_traces(auth_uid)
    user_traces = [t["trace"] for t in user_traces_data]
    traces = user_traces[-lookback:] if user_traces else []

    if len(traces) < 3:
        return {"anomalies": [], "message": "Not enough data for anomaly detection"}

    latencies = [t.get("total_latency_sec", 0) for t in traces if t.get("total_latency_sec")]
    if not latencies:
        return {"anomalies": [], "message": "No latency data available"}

    import statistics
    mean = statistics.mean(latencies)
    stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    threshold = mean + 2 * stdev

    anomalies = []
    for t in traces:
        lat = t.get("total_latency_sec", 0)
        if lat and lat > threshold:
            anomalies.append({
                "trace_id": t.get("trace_id", ""),
                "latency_sec": lat,
                "threshold": round(threshold, 2),
                "mean": round(mean, 2),
                "stddev": round(stdev, 2),
                "z_score": round((lat - mean) / stdev, 2) if stdev else 0,
                "timestamp": t.get("timestamp", ""),
            })

    return {
        "anomalies": anomalies,
        "count": len(anomalies),
        "rolling_mean": round(mean, 2),
        "rolling_stddev": round(stdev, 2),
        "threshold": round(threshold, 2),
        "sample_size": len(latencies),
    }


@router.post("/vision/analyze")
async def analyze_image_hybrid(
    image: UploadFile = File(...),
    mode: str = Form("disease"),
    crop: str = Form("crop"),
    conf_threshold: float = Form(0.25),
    user: dict = Depends(require_auth),
):
    """
    Hybrid vision analysis endpoint.
    Pipeline: YOLO11 detection → Qwen2.5-VL expert explanation
    Returns annotated image (base64), detections with bounding boxes,
    confidence scores, inference timing, and AI-generated expert insights.
    """
    if not image or not image.filename:
        return {"status": "error", "message": "No image provided"}

    try:
        image_bytes = await image.read()
    except Exception as e:
        return {"status": "error", "message": f"Failed to read image: {e}"}

    hybrid = HybridVisionService.get_instance()

    if mode == "waste":
        result = hybrid.analyze_waste(
            image_bytes, conf_threshold=conf_threshold
        )
    else:
        result = hybrid.analyze_plant_disease(
            image_bytes, crop=crop, conf_threshold=conf_threshold
        )

    result["mode"] = mode
    result["crop"] = crop
    result["user_id"] = user.get("uid", "anonymous")

    return result


@router.get("/vision/status")
def vision_status():
    """Reports availability of all vision pipeline components."""
    hybrid = HybridVisionService.get_instance()
    from services.openrouter_service import OpenRouterService
    openrouter = OpenRouterService.get_instance()

    return {
        "openrouter": {
            "available": openrouter.available,
            "model": openrouter.model,
        },
        "gemini_vision": {
            "available": gemini_service.is_available(),
        },
    }


@router.post("/auth/verify")
async def verify_auth_token(authorization: Optional[str] = Header(None)):
    """Verifies a Firebase ID token from the Authorization header."""
    if not firebase_available():
        return {"authenticated": False, "reason": "Firebase not configured"}

    if not authorization or not authorization.startswith("Bearer "):
        return {"authenticated": False, "reason": "No Bearer token provided"}

    token = authorization.replace("Bearer ", "")
    claims = verify_token_cached(token)
    if claims:
        return {
            "authenticated": True,
            "uid": claims.get("uid"),
            "email": claims.get("email"),
        }
    return {"authenticated": False, "reason": "Invalid or expired token"}
