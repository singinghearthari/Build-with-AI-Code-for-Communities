"""
SEED AI — Firebase Service
Singleton initialization for Firebase Admin SDK.
Provides Firestore client, Auth verification (with caching), and health checks.
"""
import os
import time
import hashlib
import logging
from typing import Optional

logger = logging.getLogger("FirebaseService")

_firebase_app = None
_firestore_client = None

# ── Token cache ──────────────────────────────────────────────────────────────
# Simple TTL cache for verified tokens to avoid hitting Firebase Admin SDK
# on every consecutive API request with the same token.
_token_cache: dict[str, tuple[dict, float]] = {}
_TOKEN_CACHE_TTL = 300  # 5 minutes
_TOKEN_CACHE_MAX_SIZE = 100


def _init_firebase():
    """Initializes Firebase Admin SDK once (singleton)."""
    global _firebase_app, _firestore_client

    if _firebase_app is not None:
        return

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    firebase_json = os.getenv("FIREBASE_JSON")
    
    if not firebase_json and (not creds_path or not os.path.exists(creds_path)):
        logger.warning(
            f"Firebase credentials not found. "
            "Firebase features (Auth, Firestore) will be disabled."
        )
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        import json

        if firebase_json:
            cred_dict = json.loads(firebase_json)
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate(creds_path)
            
        _firebase_app = firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")


# Initialize on import
_init_firebase()


def get_firestore_client():
    """Returns the singleton Firestore client, or None if unavailable."""
    return _firestore_client


def get_firebase_app():
    """Returns the Firebase Admin app instance."""
    return _firebase_app


def verify_token(id_token: str) -> Optional[dict]:
    """
    Verifies a Firebase ID token and returns the decoded claims.
    Returns None if verification fails or Firebase is unavailable.
    """
    if not _firebase_app:
        logger.warning("Cannot verify token: Firebase not initialized.")
        return None

    try:
        from firebase_admin import auth
        decoded = auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return None


def verify_token_cached(id_token: str) -> Optional[dict]:
    """
    Cached wrapper around verify_token.
    Uses a SHA-256 hash of the token as the cache key with a 5-minute TTL.
    Avoids hitting the Firebase Admin SDK API on every consecutive request.
    """
    global _token_cache

    # Hash the token for cache key (don't store raw tokens)
    token_hash = hashlib.sha256(id_token.encode()).hexdigest()[:16]
    now = time.time()

    # Check cache
    if token_hash in _token_cache:
        claims, cached_at = _token_cache[token_hash]
        if now - cached_at < _TOKEN_CACHE_TTL:
            return claims
        else:
            del _token_cache[token_hash]

    # Cache miss — verify with Firebase
    claims = verify_token(id_token)
    if claims is not None:
        # Evict oldest entries if cache is too large
        if len(_token_cache) >= _TOKEN_CACHE_MAX_SIZE:
            oldest_key = min(_token_cache, key=lambda k: _token_cache[k][1])
            del _token_cache[oldest_key]
        _token_cache[token_hash] = (claims, now)

    return claims


def is_available() -> bool:
    """Returns True if Firebase is initialized and ready."""
    return _firebase_app is not None


def health_check() -> dict:
    """Returns Firebase service health status."""
    status = {
        "firebase_admin": is_available(),
        "firestore": _firestore_client is not None,
    }
    if is_available():
        try:
            from firebase_admin import auth
            status["auth"] = True
        except Exception:
            status["auth"] = False
    else:
        status["auth"] = False

    return status

