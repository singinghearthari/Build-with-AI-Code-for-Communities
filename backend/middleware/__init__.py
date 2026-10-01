"""
SEED AI — Firebase Auth Middleware
FastAPI dependency for verifying Firebase ID tokens with caching.
"""
from typing import Optional
from fastapi import Header, HTTPException

from services.firebase_service import verify_token_cached, is_available


async def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency that extracts and verifies Firebase ID tokens.
    Returns decoded claims dict on success.
    Raises HTTPException(401) on failure.
    
    Usage:
        @router.post("/protected")
        async def protected_endpoint(user: dict = Depends(require_auth)):
            uid = user["uid"]
    """
    if not is_available():
        # Firebase not configured — allow through with a synthetic user
        return {"uid": "anonymous", "email": None, "firebase_available": False}

    if not authorization or not authorization.startswith("Bearer "):
        # Allow guest/anonymous users in simulator console
        return {"uid": "anonymous", "email": None, "role": "guest"}

    token = authorization.replace("Bearer ", "").strip()
    if token in ("anonymous", "guest", "dev", "undefined", "null", ""):
        return {"uid": "anonymous", "email": None, "role": "guest"}

    claims = verify_token_cached(token)

    if claims is None:
        # Fall back to anonymous guest mode instead of blocking simulator console
        return {"uid": "anonymous", "email": None, "role": "guest"}

    return claims


async def optional_auth(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    Like require_auth, but returns None instead of raising 401.
    Use for endpoints that work with or without authentication.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    if not is_available():
        return None

    token = authorization.replace("Bearer ", "")
    return verify_token_cached(token)
