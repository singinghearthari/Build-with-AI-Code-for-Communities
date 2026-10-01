"""
SEED AI — Centralized Gemini Service (Singleton v3)
Single shared client for all agents. Validates API keys at startup.
Provides health_check(), generate(), and generate_with_vision() with:
- Multi-key failover (GEMINI_API_KEY + GEMINI_API_KEY1..5)
- Rate limiting (15 RPM per key with safety margin)
- Auto key rotation on 429 / quota exhaustion
- Key cooldown: temporarily blacklists a key after 429
- Retry with exponential backoff + 429-aware wait
- Token tracking
- Async-aware rate limiting
"""
import os
import re
import time
import asyncio
import logging
import threading
from typing import Any, Optional, List
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import json

logger = logging.getLogger("GeminiService")

_clients: List[Optional[genai.Client]] = []
_api_keys: List[str] = []
_current_key_index: int = 0
_initialized: bool = False
_init_error: Optional[str] = None

_key_cooldowns: dict[int, float] = {}
_key_cooldown_lock = threading.Lock()
KEY_COOLDOWN_SECONDS = 3.0
COOLDOWN_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".gemini_cooldown_cache.json")


def _load_cooldown_cache():
    try:
        if os.path.exists(COOLDOWN_CACHE_FILE):
            with open(COOLDOWN_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                now = time.time()
                for k_str, expiry in data.items():
                    if expiry > now:
                        _key_cooldowns[int(k_str)] = expiry
            active = len([k for k, exp in _key_cooldowns.items() if exp > time.time()])
            if active > 0:
                logger.info(f"Loaded {active} active key cooldowns from persistent cache")
    except Exception as e:
        logger.debug(f"Failed to load key cooldown cache: {e}")


def _save_cooldown_cache():
    try:
        now = time.time()
        active = {str(k): exp for k, exp in _key_cooldowns.items() if exp > now}
        with open(COOLDOWN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(active, f)
    except Exception as e:
        logger.debug(f"Failed to save key cooldown cache: {e}")


_load_cooldown_cache()

API_KEY_ENV_VARS = [
    "GEMINI_API_KEY",
    "GEMINI_API_KEY1",
    "GEMINI_API_KEY2",
    "GEMINI_API_KEY3",
    "GEMINI_API_KEY4",
    "GEMINI_API_KEY5",
]

DEFAULT_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
VISION_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
]


def extract_clean_json(text: str) -> str:
    """Extract pure JSON from model response, stripping any markdown backticks or commentary."""
    if not text:
        return "{}"
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", t)
    if match:
        return match.group(0).strip()
    return t.strip()

SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_ONLY_HIGH",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]


class AsyncRateLimiter:
    """
    High-performance thread-safe sliding-window rate limiter with burst capability.
    Allows concurrent bursts for parallel agent execution while strictly enforcing
    the sliding 60-second quota across the API key pool.
    """

    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def _acquire_or_wait(self) -> float:
        """Atomically checks if slot is available; returns seconds to wait or 0 if slot acquired."""
        with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            if len(self._timestamps) >= self.max_per_minute:
                oldest = self._timestamps[0]
                return max(60.0 - (now - oldest) + 0.05, 0.05)
            self._timestamps.append(now)
            return 0.0

    async def wait_if_needed(self):
        """Async wait — never blocks the event loop during sleep."""
        while True:
            wait = self._acquire_or_wait()
            if wait <= 0:
                break
            logger.info(f"Rate limiter: waiting {wait:.2f}s (at {self.max_per_minute} RPM limit)")
            await asyncio.sleep(wait)

    def sync_wait_if_needed(self):
        """Sync wait — safe to call from any thread."""
        while True:
            wait = self._acquire_or_wait()
            if wait <= 0:
                break
            logger.info(f"Rate limiter: waiting {wait:.2f}s (sync, at {self.max_per_minute} RPM limit)")
            time.sleep(wait)

_rate_limiter = AsyncRateLimiter(max_per_minute=90)
_gemini_concurrency = threading.Semaphore(10)
_gemini_concurrency_max = 10


def _check_groq_available() -> bool:
    try:
        import groq  # noqa: F401
        return True
    except ImportError:
        return False


def _check_g4f_available() -> bool:
    try:
        import g4f  # noqa: F401
        return True
    except ImportError:
        return False


# Cache whether fallback libs are importable (checked once at startup)
_groq_available: bool = _check_groq_available()
_g4f_available: bool = _check_g4f_available()


def groq_available() -> bool:
    return _groq_available


def g4f_available() -> bool:
    return _g4f_available


def set_rate_limit(max_per_minute: int):
    global _rate_limiter
    _rate_limiter = AsyncRateLimiter(max_per_minute=max_per_minute)
    logger.info(f"Rate limit updated to {max_per_minute} RPM")


def set_concurrency(max_concurrent: int):
    global _gemini_concurrency, _gemini_concurrency_max
    _gemini_concurrency = threading.Semaphore(max_concurrent)
    _gemini_concurrency_max = max_concurrent
    logger.info(f"Gemini concurrency set to {max_concurrent}")


def _collect_api_keys() -> List[str]:
    keys = []
    for env_var in API_KEY_ENV_VARS:
        key = os.getenv(env_var, "")
        if key and not key.startswith("AIzaSyBk_placeholder"):
            if key not in keys:
                keys.append(key)
                logger.info(f"Loaded key from {env_var} (first 12 chars: {key[:12]}...)")
            else:
                logger.info(f"Skipping duplicate key from {env_var}")
    logger.info(f"Collected {len(keys)} unique Gemini API keys")
    return keys


def _is_key_on_cooldown(index: int) -> bool:
    with _key_cooldown_lock:
        expiry = _key_cooldowns.get(index)
        now = time.time()
        if expiry and now < expiry:
            return True
        if expiry:
            del _key_cooldowns[index]
            _save_cooldown_cache()
        return False


def _mark_key_cooldown(index: int, duration: float = KEY_COOLDOWN_SECONDS):
    if index < 0:
        return
    with _key_cooldown_lock:
        _key_cooldowns[index] = time.time() + duration
        _save_cooldown_cache()


def _get_available_client_indices() -> List[int]:
    return [
        i for i, c in enumerate(_clients)
        if c is not None and not _is_key_on_cooldown(i)
    ]


_key_rotation_lock = threading.Lock()
_round_robin_idx = 0


def _rotate_key():
    global _current_key_index, _round_robin_idx
    with _key_rotation_lock:
        valid_indices = [i for i, c in enumerate(_clients) if c is not None]
        if not valid_indices:
            return
        available = [i for i in valid_indices if not _is_key_on_cooldown(i)]
        if not available:
            return
        _round_robin_idx = (_round_robin_idx + 1) % len(available)
        _current_key_index = available[_round_robin_idx]


def init_gemini() -> None:
    global _clients, _api_keys, _initialized, _init_error

    if _initialized:
        return

    _api_keys = _collect_api_keys()
    if not _api_keys:
        _init_error = (
            "No valid GEMINI_API_KEY environment variables found. "
            "Set GEMINI_API_KEY (or GEMINI_API_KEY1..GEMINI_API_KEY4 for failover)."
        )
        logger.error(_init_error)
        _initialized = True
        return

    _clients = []
    for idx, key in enumerate(_api_keys):
        try:
            http_opts = types.HttpOptions(
                timeout=10000,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            client = genai.Client(api_key=key, http_options=http_opts)
            _clients.append(client)
            logger.info(f"Gemini client {idx + 1}/{len(_api_keys)} initialized (key: {key[:12]}...).")
        except Exception as e:
            logger.warning(f"Gemini client {idx + 1} initialization failed for key {key[:12]}...: {e}")
            _clients.append(None)

    healthy = sum(1 for c in _clients if c is not None)
    if healthy == 0:
        _init_error = "All Gemini clients failed to initialize."
    else:
        logger.info(f"{healthy}/{len(_clients)} Gemini clients ready.")
        _init_error = None
        safe_rpm = max(healthy * 15, 60)
        set_rate_limit(safe_rpm)
        set_concurrency(max(healthy, 6))
        logger.info(f"Rate limit set to {safe_rpm} RPM across {healthy} keys")

    _initialized = True


def get_client_with_index() -> Tuple[Optional[genai.Client], int]:
    if not _initialized:
        init_gemini()
    if not _clients:
        return None, -1

    global _current_key_index, _round_robin_idx
    with _key_rotation_lock:
        valid_indices = [i for i, c in enumerate(_clients) if c is not None]
        if not valid_indices:
            return None, -1

        available = [i for i in valid_indices if not _is_key_on_cooldown(i)]
        if not available:
            return None, -1

        _round_robin_idx = (_round_robin_idx + 1) % len(available)
        chosen_idx = available[_round_robin_idx]
        _current_key_index = chosen_idx
        return _clients[chosen_idx], chosen_idx


def get_client() -> Optional[genai.Client]:
    client, _ = get_client_with_index()
    return client


def get_current_key_index() -> int:
    return _current_key_index


def is_available() -> bool:
    if not _initialized:
        init_gemini()
    return any(c is not None for c in _clients)


def get_init_error() -> Optional[str]:
    if not _initialized:
        init_gemini()
    return _init_error


def health_check() -> dict:
    if not _initialized:
        init_gemini()

    client = get_client()
    status: dict[str, Any] = {
        "available": client is not None,
        "model": DEFAULT_MODEL,
        "error": _init_error,
        "active_key_index": _current_key_index,
        "total_keys": len(_api_keys),
    }

    if client is not None:
        try:
            _rate_limiter.sync_wait_if_needed()
            start = time.time()
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents="Reply with exactly: OK",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=5,
                ),
            )
            latency_ms = (time.time() - start) * 1000
            status["ping"] = "ok"
            status["ping_latency_ms"] = round(latency_ms, 1)
        except Exception as e:
            status["ping"] = "failed"
            status["ping_error"] = str(e)
            _rotate_key()

    return status


class GenerationResult:
    """Wraps a Gemini response with extracted metadata."""

    def __init__(
        self,
        text: str,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        model: str = DEFAULT_MODEL,
    ):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.total_tokens = prompt_tokens + output_tokens
        self.latency_ms = latency_ms
        self.model = model


def _extract_retry_delay(error: Exception) -> Optional[float]:
    msg = str(error)
    match = re.search(r"retry in (\d+(?:\.\d+)?)", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _handle_rate_limit_error(error: Exception) -> None:
    delay = _extract_retry_delay(error)
    if delay and delay > 0:
        wait = min(delay + 2.0, 90.0)
        logger.warning(f"429 Rate limited. Waiting {wait:.1f}s before retry...")
        time.sleep(wait)


GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


def _groq_fallback(
    prompt: Any,
    schema: Optional[Any] = None,
    models: list = GROQ_MODELS,
) -> Optional[GenerationResult]:
    """Fallback generator using Groq when Gemini limits are reached.
    Tries multiple models sequentially. Returns None if all fail."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set — skipping Groq fallback")
        return None
    if not _groq_available:
        logger.warning("groq SDK not installed — skipping Groq fallback")
        return None

    prompt_text = prompt
    if isinstance(prompt, list):
        prompt_text = next((item for item in prompt if isinstance(item, str)), "Please analyze the input.")

    messages = []
    if schema:
        if hasattr(schema, "model_json_schema"):
            schema_dict = schema.model_json_schema()
            schema_str = json.dumps(schema_dict, indent=2)
            sys_msg = (
                "You must return ONLY a raw valid JSON object. "
                "Do not use markdown code blocks like ```json. "
                f"Your JSON must strictly match this schema:\n{schema_str}"
            )
            messages.append({"role": "system", "content": sys_msg})

    messages.append({"role": "user", "content": str(prompt_text)})

    for model in models:
        for attempt in range(2):
            logger.info(f"Using high-speed Groq ({model}) fallback (attempt {attempt + 1})...")
            try:
                from groq import Groq
                client = Groq(api_key=groq_api_key)
                start = time.time()
                extra_kwargs = {}
                if schema:
                    extra_kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=800,
                    **extra_kwargs
                )
                result_text = response.choices[0].message.content or ""
                latency_ms = (time.time() - start) * 1000

                if schema:
                    result_text = result_text.strip("` \n")
                    if result_text.startswith("json\n"):
                        result_text = result_text[5:]
                    import re as _re
                    json_match = _re.search(r'\{.*\}', result_text, _re.DOTALL)
                    if json_match:
                        result_text = json_match.group(0)

                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return GenerationResult(
                    text=result_text,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    model=f"{model} (groq fallback)"
                )
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate_limit" in err_str.lower() or "tokens" in err_str.lower()) and attempt == 0:
                    import re as _re
                    delay_match = _re.search(r'try again in ([\d\.]+)(m?s)', err_str)
                    delay = 0.8
                    if delay_match:
                        val = float(delay_match.group(1))
                        unit = delay_match.group(2)
                        delay = val / 1000.0 if unit == "ms" else val
                    wait_time = min(max(delay + 0.1, 0.5), 2.0)
                    logger.info(f"Groq {model} rate limited, waiting {wait_time:.2f}s before fast retry...")
                    time.sleep(wait_time)
                    continue
                logger.warning(f"Groq {model} failed: {e}")
                break

    logger.error("All Groq models exhausted")
    return None


def _openrouter_fallback(
    prompt: Any,
    schema: Optional[Any] = None,
) -> Optional[GenerationResult]:
    """Fast secondary fallback via OpenRouter API."""
    try:
        try:
            from services.openrouter_service import OpenRouterService
        except ImportError:
            from backend.services.openrouter_service import OpenRouterService

        ors = OpenRouterService.get_instance()
        if ors.available:
            res = ors.generate(prompt, schema=schema)
            if res and res.get("text"):
                raw_text = res["text"]
                clean_text = extract_clean_json(raw_text) if schema else raw_text
                return GenerationResult(
                    text=clean_text,
                    prompt_tokens=res.get("prompt_tokens", 0),
                    output_tokens=res.get("output_tokens", 0),
                    latency_ms=res.get("latency_ms", 0.0),
                    model=res.get("model", "OpenRouter fallback")
                )
    except Exception as e:
        logger.warning(f"OpenRouter fallback attempt failed: {e}")
    return None


def _fallback_generate(
    prompt: Any,
    schema: Optional[Any] = None,
    model: str = "fallback"
) -> Optional[GenerationResult]:
    """Fast non-blocking fallback generator. Never hangs."""
    return None


def _run_fallback_chain(
    prompt: Any,
    schema: Optional[Any] = None,
) -> Optional[GenerationResult]:
    """Try fallback providers in order: OpenRouter → Groq. Returns first success or None instantly."""
    result = _openrouter_fallback(prompt, schema=schema)
    if result is not None:
        return result
    result = _groq_fallback(prompt, schema=schema)
    if result is not None:
        return result
    return None


async def _async_openrouter_fallback(
    prompt: Any,
    schema: Optional[Any] = None,
) -> Optional[GenerationResult]:
    """Fast non-blocking secondary fallback via OpenRouter API."""
    try:
        try:
            from services.openrouter_service import OpenRouterService
        except ImportError:
            from backend.services.openrouter_service import OpenRouterService

        ors = OpenRouterService.get_instance()
        if ors.available:
            res = await ors.async_generate(prompt, schema=schema)
            if res and res.get("text"):
                raw_text = res["text"]
                clean_text = extract_clean_json(raw_text) if schema else raw_text
                return GenerationResult(
                    text=clean_text,
                    prompt_tokens=res.get("prompt_tokens", 0),
                    output_tokens=res.get("output_tokens", 0),
                    latency_ms=res.get("latency_ms", 0.0),
                    model=res.get("model", "OpenRouter fallback")
                )
    except Exception as e:
        logger.warning(f"OpenRouter async fallback attempt failed: {e}")
    return None


async def _async_run_fallback_chain(
    prompt: Any,
    schema: Optional[Any] = None,
) -> Optional[GenerationResult]:
    """Try async fallback providers: OpenRouter → Groq. Returns first success or None instantly."""
    result = await _async_openrouter_fallback(prompt, schema=schema)
    if result is not None:
        return result
    result = await asyncio.to_thread(_groq_fallback, prompt, schema=schema)
    if result is not None:
        return result
    return None


def generate(
    prompt: Any,
    schema: Optional[Any] = None,
    temperature: float = 0.2,
    model: str = DEFAULT_MODEL,
    tools: Optional[list] = None,
) -> GenerationResult:
    """
    Core generation call with rate limiting, multi-key failover, token tracking, and timing.
    """
    if not _initialized:
        init_gemini()

    # Fast bypass: if all Gemini keys are on cooldown or unavailable, route directly to fallback with zero semaphore lock
    client, key_idx = get_client_with_index()
    if client is None or not _clients or all(c is None for c in _clients):
        logger.info("All Gemini keys cooling down or exhausted, using fallback chain immediately...")
        fb = _run_fallback_chain(prompt, schema=schema)
        if fb is not None:
            return fb
        raise ValueError(
            f"Gemini client not available: {_init_error or 'All keys on cooldown'}"
        )

    last_error = None
    key_attempts = 0
    max_key_attempts = max(len(_clients), 1) * 2

    _gemini_concurrency.acquire()
    try:
        while key_attempts < max_key_attempts:
            if client is None:
                client, key_idx = get_client_with_index()
                if client is None:
                    break

            _rate_limiter.sync_wait_if_needed()

            config = types.GenerateContentConfig(
                temperature=temperature,
                safety_settings=SAFETY_SETTINGS,
            )
            if schema:
                config.response_mime_type = "application/json"
                config.response_schema = schema
            if tools:
                config.tools = tools

            models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
            for target_model in models_to_try:
                try:
                    start = time.time()
                    response = client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                        config=config,
                    )
                    latency_ms = (time.time() - start) * 1000

                    prompt_tokens = 0
                    output_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage = response.usage_metadata
                        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

                    raw_text = response.text if hasattr(response, "text") else ""
                    clean_text = extract_clean_json(raw_text) if schema else raw_text

                    return GenerationResult(
                        text=clean_text,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        model=target_model,
                    )
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    if any(err_kw in error_msg.lower() for err_kw in ("404", "not_found", "503", "unavailable", "500", "502", "high demand", "overloaded", "capacity", "no capacity")):
                        logger.info(f"Model {target_model} temporary error ({error_msg[:60]}), trying next model...")
                        continue
                    break

            key_attempts += 1
            key_label = f"key #{key_idx + 1}" if key_idx >= 0 else "unknown key"

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                cooldown_dur = 60.0 if ("limit: 20" in error_msg or "per day" in error_msg.lower()) else 5.0
                logger.warning(f"Rate limit on {key_label}, cooling down for {cooldown_dur}s")
                _mark_key_cooldown(key_idx, duration=cooldown_dur)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                logger.warning("All Gemini keys exhausted on rate limit, switching to fallback chain...")
                break
            elif any(err_kw in error_msg.lower() for err_kw in ("503", "unavailable", "high demand", "overloaded", "500", "502", "capacity", "no capacity")):
                logger.warning(f"Temporary server overload on {key_label} ({error_msg[:60]}), cooling down for 3s and rotating...")
                _mark_key_cooldown(key_idx, duration=3.0)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
            else:
                logger.warning(f"Non-rate-limit error on {key_label}: {error_msg[:120]}")
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
    finally:
        _gemini_concurrency.release()

    fb = _run_fallback_chain(prompt, schema=schema)
    if fb is not None:
        return fb
    raise last_error or ValueError("All LLM providers exhausted")


async def async_generate(
    prompt: Any,
    schema: Optional[Any] = None,
    temperature: float = 0.2,
    model: str = DEFAULT_MODEL,
    tools: Optional[list] = None,
) -> GenerationResult:
    """
    Fully async version of generate().
    Bypasses semaphore and rate limiter when keys are exhausted to allow concurrent fallback processing.
    """
    if not _initialized:
        init_gemini()

    # Fast bypass: if all Gemini keys are on cooldown or unavailable, route directly to async fallback without blocking semaphore
    client, key_idx = get_client_with_index()
    if client is None or not _clients or all(c is None for c in _clients):
        logger.info("All Gemini keys cooling down or exhausted, using async fallback chain immediately...")
        fb = await _async_run_fallback_chain(prompt, schema=schema)
        if fb is not None:
            return fb
        raise ValueError(
            f"Gemini client not available: {_init_error or 'All keys on cooldown'}"
        )

    last_error = None
    key_attempts = 0
    max_key_attempts = max(len(_clients), 1) * 2

    await asyncio.to_thread(_gemini_concurrency.acquire)
    try:
        while key_attempts < max_key_attempts:
            if client is None:
                client, key_idx = get_client_with_index()
                if client is None:
                    break

            await _rate_limiter.wait_if_needed()

            config = types.GenerateContentConfig(
                temperature=temperature,
                safety_settings=SAFETY_SETTINGS,
            )
            if schema:
                config.response_mime_type = "application/json"
                config.response_schema = schema
            if tools:
                config.tools = tools

            models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
            for target_model in models_to_try:
                try:
                    start = time.time()
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=target_model,
                        contents=prompt,
                        config=config,
                    )
                    latency_ms = (time.time() - start) * 1000

                    prompt_tokens = 0
                    output_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage = response.usage_metadata
                        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

                    raw_text = response.text if hasattr(response, "text") else ""
                    clean_text = extract_clean_json(raw_text) if schema else raw_text

                    return GenerationResult(
                        text=clean_text,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        model=target_model,
                    )
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    if any(err_kw in error_msg.lower() for err_kw in ("404", "not_found", "503", "unavailable", "500", "502", "high demand", "overloaded", "capacity", "no capacity")):
                        logger.info(f"Model {target_model} temporary error ({error_msg[:60]}), trying next model...")
                        continue
                    break

            key_attempts += 1
            key_label = f"key #{key_idx + 1}" if key_idx >= 0 else "unknown key"

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                cooldown_dur = 60.0 if ("limit: 20" in error_msg or "per day" in error_msg.lower()) else 5.0
                logger.warning(f"Rate limit on {key_label}, cooling down for {cooldown_dur}s")
                _mark_key_cooldown(key_idx, duration=cooldown_dur)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                logger.warning("All Gemini keys exhausted on rate limit, switching to async fallback chain...")
                break
            elif any(err_kw in error_msg.lower() for err_kw in ("503", "unavailable", "high demand", "overloaded", "500", "502", "capacity", "no capacity")):
                logger.warning(f"Temporary server overload on {key_label} ({error_msg[:60]}), cooling down for 3s and rotating...")
                _mark_key_cooldown(key_idx, duration=3.0)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
            else:
                logger.warning(f"Non-rate-limit error on {key_label}: {error_msg[:120]}")
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
    finally:
        await asyncio.to_thread(_gemini_concurrency.release)

    fb = await _async_run_fallback_chain(prompt, schema=schema)
    if fb is not None:
        return fb
    raise last_error or ValueError("All LLM providers exhausted")


async def async_generate_with_vision(
    contents: list,
    schema: Optional[Any] = None,
    temperature: float = 0.1,
    model: str = DEFAULT_MODEL,
) -> GenerationResult:
    """
    Fully async version of generate_with_vision().
    Uses async rate limiter directly.
    """
    if not _initialized:
        init_gemini()

    client, key_idx = get_client_with_index()
    if client is None or not _clients or all(c is None for c in _clients):
        logger.warning("All Gemini keys unavailable/cooldown for vision, trying async fallback chain...")
        fb = await _async_run_fallback_chain(contents, schema=schema)
        if fb is not None:
            return fb
        raise ValueError(
            f"Gemini client not available: {_init_error or 'All keys on cooldown'}"
        )

    last_error = None
    key_attempts = 0
    max_key_attempts = max(len(_clients), 1) * 2

    await asyncio.to_thread(_gemini_concurrency.acquire)
    try:
        while key_attempts < max_key_attempts:
            if client is None:
                client, key_idx = get_client_with_index()
                if client is None:
                    break

            await _rate_limiter.wait_if_needed()

            config = types.GenerateContentConfig(
                temperature=temperature,
                safety_settings=SAFETY_SETTINGS,
            )
            if schema:
                config.response_mime_type = "application/json"
                config.response_schema = schema

            models_to_try = [model] + [m for m in VISION_MODELS if m != model]
            for target_model in models_to_try:
                try:
                    start = time.time()
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=target_model,
                        contents=contents,
                        config=config,
                    )
                    latency_ms = (time.time() - start) * 1000

                    prompt_tokens = 0
                    output_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage = response.usage_metadata
                        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

                    raw_text = response.text if hasattr(response, "text") else ""
                    clean_text = extract_clean_json(raw_text) if schema else raw_text

                    return GenerationResult(
                        text=clean_text,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        model=target_model,
                    )
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    if any(err_kw in error_msg.lower() for err_kw in ("404", "not_found", "503", "unavailable", "500", "502", "high demand", "overloaded", "capacity", "no capacity")):
                        logger.info(f"Vision model {target_model} temporary error ({error_msg[:60]}), trying next model...")
                        continue
                    break

            key_attempts += 1
            key_label = f"key #{key_idx + 1}" if key_idx >= 0 else "unknown key"

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                cooldown_dur = 60.0 if ("limit: 20" in error_msg or "per day" in error_msg.lower()) else 5.0
                logger.warning(f"Rate limit on {key_label}, cooling down for {cooldown_dur}s")
                _mark_key_cooldown(key_idx, duration=cooldown_dur)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
            elif any(err_kw in error_msg.lower() for err_kw in ("503", "unavailable", "high demand", "overloaded", "500", "502", "capacity", "no capacity")):
                logger.warning(f"Temporary server overload on {key_label} ({error_msg[:60]}), cooling down for 3s and rotating...")
                _mark_key_cooldown(key_idx, duration=3.0)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
            else:
                logger.warning(f"Non-rate-limit error on {key_label}: {error_msg[:120]}")
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
    finally:
        await asyncio.to_thread(_gemini_concurrency.release)

    fb = await _async_run_fallback_chain(contents, schema=schema)
    if fb is not None:
        return fb
    raise last_error or ValueError("All API keys exhausted")


def generate_with_vision(
    contents: list,
    schema: Optional[Any] = None,
    temperature: float = 0.1,
    model: str = DEFAULT_MODEL,
) -> GenerationResult:
    """
    Vision-specific generation (multimodal content list).
    Same rate limiting, multi-key failover, and tracking as generate().
    """
    if not _initialized:
        init_gemini()

    client, key_idx = get_client_with_index()
    if client is None or not _clients or all(c is None for c in _clients):
        logger.warning("All Gemini keys unavailable/cooldown for vision, trying fallback chain...")
        fb = _run_fallback_chain(contents, schema=schema)
        if fb is not None:
            return fb
        raise ValueError(
            f"Gemini client not available: {_init_error or 'All keys on cooldown'}"
        )

    last_error = None
    key_attempts = 0
    max_key_attempts = max(len(_clients), 1) * 2

    _gemini_concurrency.acquire()
    try:
        while key_attempts < max_key_attempts:
            if client is None:
                client, key_idx = get_client_with_index()
                if client is None:
                    break

            _rate_limiter.sync_wait_if_needed()

            config = types.GenerateContentConfig(
                temperature=temperature,
                safety_settings=SAFETY_SETTINGS,
            )
            if schema:
                config.response_mime_type = "application/json"
                config.response_schema = schema

            models_to_try = [model] + [m for m in VISION_MODELS if m != model]
            for target_model in models_to_try:
                try:
                    start = time.time()
                    response = client.models.generate_content(
                        model=target_model,
                        contents=contents,
                        config=config,
                    )
                    latency_ms = (time.time() - start) * 1000

                    prompt_tokens = 0
                    output_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage = response.usage_metadata
                        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

                    raw_text = response.text if hasattr(response, "text") else ""
                    clean_text = extract_clean_json(raw_text) if schema else raw_text

                    return GenerationResult(
                        text=clean_text,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        model=target_model,
                    )
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    if any(err_kw in error_msg.lower() for err_kw in ("404", "not_found", "503", "unavailable", "500", "502", "high demand", "overloaded", "capacity", "no capacity")):
                        logger.info(f"Vision model {target_model} temporary error ({error_msg[:60]}), trying next model...")
                        continue
                    break

            key_attempts += 1
            key_label = f"key #{key_idx + 1}" if key_idx >= 0 else "unknown key"

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                cooldown_dur = 60.0 if ("limit: 20" in error_msg or "per day" in error_msg.lower()) else 5.0
                logger.warning(f"Rate limit on {key_label}, cooling down for {cooldown_dur}s")
                _mark_key_cooldown(key_idx, duration=cooldown_dur)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
            elif any(err_kw in error_msg.lower() for err_kw in ("503", "unavailable", "high demand", "overloaded", "500", "502", "capacity", "no capacity")):
                logger.warning(f"Temporary server overload on {key_label} ({error_msg[:60]}), cooling down for 3s and rotating...")
                _mark_key_cooldown(key_idx, duration=3.0)
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
            else:
                logger.warning(f"Non-rate-limit error on {key_label}: {error_msg[:120]}")
                _rotate_key()
                client, key_idx = get_client_with_index()
                if client is not None:
                    continue
                break
    finally:
        _gemini_concurrency.release()

    fb = _run_fallback_chain(contents, schema=schema)
    if fb is not None:
        return fb
    raise last_error or ValueError("All API keys exhausted")
