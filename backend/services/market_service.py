import os
import time
import logging
from typing import Dict, Tuple
import requests

logger = logging.getLogger("MarketService")

_MARKET_CACHE: Dict[str, Tuple[dict, float]] = {}
MARKET_CACHE_TTL = 1800  # 30 minutes cache
_CIRCUIT_OPEN_UNTIL = 0.0
CIRCUIT_COOLDOWN_SECONDS = 300.0  # 5 minutes if endpoint is down


class MarketService:
    """
    Production market price service with caching, 2s fail-fast timeout,
    and automatic circuit breaker to prevent blocking the agent swarm.
    """

    def __init__(self):
        self.api_url = os.getenv(
            "AGMARKNET_API_URL",
            "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
        )
        self.api_key = os.getenv("DATA_GOV_IN_API_KEY")

    def get_price(self, crop: str, state: str = None) -> dict:
        global _CIRCUIT_OPEN_UNTIL
        now = time.time()

        if not self.api_key:
            return {"error": "DATA_GOV_IN_API_KEY not set", "crop": crop, "source": "none"}

        # Circuit breaker: if API was recently unreachable, fail immediately
        if now < _CIRCUIT_OPEN_UNTIL:
            logger.info("Market API circuit breaker active — returning instant fallback")
            return {"crop": crop, "price": None, "trend": "no data available", "source": "offline_fallback"}

        cache_key = f"{(crop or '').strip().lower()}:{(state or '').strip().lower()}"
        if cache_key in _MARKET_CACHE:
            cached_data, cached_time = _MARKET_CACHE[cache_key]
            if now - cached_time < MARKET_CACHE_TTL:
                logger.info(f"Market cache hit for '{crop}' in '{state}'")
                return cached_data

        try:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "filters[commodity]": crop,
            }
            if state:
                params["filters[state]"] = state

            # Fast 2.0s timeout with single attempt — never blocks pipeline
            resp = requests.get(self.api_url, params=params, timeout=2.0)
            resp.raise_for_status()
            data = resp.json()

            records = data.get("records", [])
            if not records:
                res = {"crop": crop, "price": None, "trend": "no data available", "source": "data.gov.in"}
                _MARKET_CACHE[cache_key] = (res, now)
                return res

            avg_price = sum(float(r["modal_price"]) for r in records) / len(records)
            res = {
                "crop": crop,
                "price_per_quintal": round(avg_price, 2),
                "num_records": len(records),
                "source": "data.gov.in",
            }
            _MARKET_CACHE[cache_key] = (res, now)
            return res

        except Exception as e:
            # Trip circuit breaker for 5 minutes so subsequent calls are instant
            _CIRCUIT_OPEN_UNTIL = time.time() + CIRCUIT_COOLDOWN_SECONDS
            logger.warning(f"Market API unavailable ({e}). Circuit breaker opened for 5m.")
            res = {"crop": crop, "price": None, "trend": "no data available", "source": "fallback"}
            _MARKET_CACHE[cache_key] = (res, now)
            return res
