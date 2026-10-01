"""
SEED AI — Backend Entry Point (Production)
Validates configuration at startup. Never silently proceeds with broken config.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables FIRST, before anything reads them
# Load backend/.env first (local overrides), then root .env (shared config)
load_dotenv()
_root_env = Path(__file__).resolve().parent.parent / ".env"
if _root_env.exists():
    load_dotenv(_root_env, override=False)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from api.tts_routes import router as tts_router
from utils.logger import setup_logger
from services import gemini_service
from services.event_bus import EventBus
from services.persistence_handler import FirestorePersistenceHandler
from agents.memory_agent import MemoryAgent
import uvicorn

# Setup structured JSON logging
logger = setup_logger()
logger.info("SEED AI Backend starting...")

# ── Startup validation ──────────────────────────────────────────────────
def _validate_config():
    """Validates critical configuration at startup. Logs warnings/errors."""

    # Required: Gemini API Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("CRITICAL: GEMINI_API_KEY is not set. All Gemini calls will fail.")
    elif gemini_key.startswith("AIzaSyBk_placeholder"):
        logger.error(
            "CRITICAL: GEMINI_API_KEY contains a placeholder value. "
            "Replace it with a valid key in .env file."
        )
    else:
        logger.info("GEMINI_API_KEY: configured")

    # Required: Firebase credentials
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Firebase disabled.")
    elif not os.path.exists(creds):
        logger.warning(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {creds}")
    else:
        logger.info("GOOGLE_APPLICATION_CREDENTIALS: configured")

    # Optional: Weather API
    if os.getenv("OPENWEATHER_API_KEY"):
        logger.info("OPENWEATHER_API_KEY: configured")
    else:
        logger.warning("OPENWEATHER_API_KEY not set. Weather agent will use Gemini fallback.")

    # Optional: Market API
    if os.getenv("DATA_GOV_IN_API_KEY"):
        logger.info("DATA_GOV_IN_API_KEY: configured")
    else:
        logger.warning("DATA_GOV_IN_API_KEY not set. Market agent will use Gemini fallback.")

    # Initialize Gemini singleton
    gemini_service.init_gemini()
    if gemini_service.is_available():
        logger.info("Gemini client: READY")
    else:
        logger.error(f"Gemini client: UNAVAILABLE — {gemini_service.get_init_error()}")


_validate_config()

# ── EventBus lifecycle ───────────────────────────────────────────────────
_memory_agent = MemoryAgent()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Startup: initialize EventBus. Shutdown: gracefully stop it."""
    from services.bus import init_bus
    _bus = init_bus(_memory_agent, flush_interval=2.0, batch_size=50)
    await _bus.start()
    logger.info("EventBus initialized and started")
    yield
    await _bus.shutdown()
    logger.info("EventBus shut down")


# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(
    title="SEED AI API",
    description="Backend API for SEED AI — Smart Ecosystem for Enhanced Agricultural Decisions",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "https://seed-ai-capstone-project.web.app",
        "https://seed-ai-capstone-project.firebaseapp.com"
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.include_router(tts_router, prefix="/api")


@app.get("/")
def root():
    return {
        "status": "SEED AI Backend Active",
        "version": "2.0.0",
        "gemini_available": gemini_service.is_available(),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting SEED AI backend on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
