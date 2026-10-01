"""
SEED AI — Orchestrator Agent (Production v4 — Async Optimized)
Deterministic routing, concurrent Tier 1 + immediate fusion, Tier 2 truly fire-and-forget.
TTFT < 400ms, total pipeline < 2.5s. Async rate limiter + result cache.
"""
import asyncio
import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set
import traceback

from google.genai import types

from .base_agent import BaseAgent
from .vision_agent import VisionAgent
from .weather_agent import WeatherAgent
from .market_intelligence_agent import MarketIntelligenceAgent
from .budget_planning_agent import BudgetPlanningAgent
from .crop_knowledge_agent import CropKnowledgeAgent
from .government_scheme_agent import GovernmentSchemeAgent
from .task_planning_agent import TaskPlanningAgent
from .crop_prediction_agent import CropPredictionAgent
from .disease_prediction_agent import DiseasePredictionAgent
from .waste_to_wealth_agent import WasteToWealthAgent
from .soil_nutrient_agent import SoilNutrientAgent
from .entomologist_agent import EntomologistAgent
from .irrigation_agent import IrrigationAgent
from .memory_agent import MemoryAgent
from models.schemas import AgentResult, AgentStatus, FusionResult, LLMFusionSchema, ExecutionReport, ProgressEvent
from services.storage_service import StorageService
from services import gemini_service
from services.bus import get_bus, is_initialized as bus_is_initialized
from evaluation.benchmark import AgentEvaluator


# ── Result Cache ──────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}
_cache_ttl: Dict[str, float] = {}
CACHE_TTL_SECONDS = 300
MAX_CACHE_SIZE = 100

def _cache_key(context: Dict[str, Any]) -> str:
    """Generate a cache key from the relevant context fields."""
    key_parts = (
        str(context.get("text_query") or ""),
        str(context.get("location") or ""),
        str(context.get("crop") or ""),
        str(context.get("budget", 0)),
        str(context.get("execution_mode") or "auto"),
        str(context.get("specific_agent") or ""),
        str(bool(context.get("image_bytes") or context.get("image_base64"))),
    )
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()

def _get_cached(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    expiry = _cache_ttl.get(key)
    if entry and expiry and time.time() < expiry:
        return entry
    if key in _cache:
        del _cache[key]
        del _cache_ttl[key]
    return None

def _set_cache(key: str, value: Any):
    if len(_cache) >= MAX_CACHE_SIZE:
        oldest = min(_cache_ttl, key=lambda k: _cache_ttl[k])
        del _cache[oldest]
        del _cache_ttl[oldest]
    _cache[key] = value
    _cache_ttl[key] = time.time() + CACHE_TTL_SECONDS


# ── Thread pool for async offloading ──────────────────────────────────────
_agent_executor = None


AGENT_TOOL_DECLARATIONS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="run_vision_analysis",
            description="Analyze a crop leaf image using Gemini Vision to detect diseases, assess severity, and recommend immediate actions. Only call if the farmer uploaded an image.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why vision analysis is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_weather_assessment",
            description="Fetch live weather data and assess agricultural risk including rain probability, spray safety, and irrigation needs.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why weather data is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_market_analysis",
            description="Get current market prices and selling recommendations for the farmer's crop.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why market analysis is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_budget_planning",
            description="Plan treatment costs within the farmer's budget, recommending cheapest and best-value options.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why budget planning is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_crop_knowledge_lookup",
            description="Retrieve disease information, treatment protocols, and best practices from the agricultural knowledge base.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why crop knowledge is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_government_scheme_search",
            description="Find eligible government subsidies, insurance schemes, and financial aid programs for the farmer.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why scheme search is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_task_scheduling",
            description="Generate a 7-day farming schedule with prioritized daily tasks based on all available context.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why task scheduling is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_crop_prediction",
            description="Predict optimal crops to grow based on location, season, soil conditions, and budget. Use when the farmer asks what to plant next or wants crop recommendations.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why crop prediction is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_disease_prediction",
            description="Predict potential crop diseases BEFORE symptoms appear based on weather, crop type, and regional disease history. Use for preventive disease management.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why disease prediction is needed"),
                },
                required=["reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_waste_to_wealth_analysis",
            description="Identify revenue opportunities from agricultural waste such as crop residues, husks, stalks, and other byproducts. Suggests composting, biochar, biofuel, and other conversion methods.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(type="STRING", description="Why waste-to-wealth analysis is needed"),
                },
                required=["reason"],
            ),
        ),
    ])
]

FUNCTION_TO_AGENT = {
    "run_vision_analysis": "vision",
    "run_weather_assessment": "weather",
    "run_market_analysis": "market",
    "run_budget_planning": "budget",
    "run_crop_knowledge_lookup": "crop_knowledge",
    "run_government_scheme_search": "government_schemes",
    "run_task_scheduling": "task_planning",
    "run_crop_prediction": "crop_prediction",
    "run_disease_prediction": "disease_prediction",
    "run_waste_to_wealth_analysis": "waste_to_wealth",
}

# Tier 1: core agents needed for initial fusion — executed concurrently
TIER1_CORE = {"crop_knowledge", "weather", "soil_nutrient", "budget"}

# Tier 2: late-enrichment agents — fire-and-forget after fusion
TIER2_ENRICHMENT = {"government_schemes", "waste_to_wealth", "task_planning"}

NON_CRITICAL_AGENTS = {"market", "government_schemes", "task_planning", "crop_prediction", "disease_prediction", "waste_to_wealth"}

MARKET_KEYWORDS = {"price", "mandi", "market", "cost", "sell"}

EMOJI_MAP = {
    "vision": "🌿", "weather": "🌦", "market": "📈",
    "budget": "💰", "crop_knowledge": "📚",
    "government_schemes": "🏛", "task_planning": "📅",
    "crop_prediction": "🌱", "disease_prediction": "🔬",
    "waste_to_wealth": "♻️",
    "soil_nutrient": "🧪", "entomologist": "🐛", "irrigation": "💧",
}

AGENT_TIMEOUT_SECONDS = 10


class OrchestratorAgent(BaseAgent):

    def __init__(self):
        super().__init__("Orchestrator")
        self.vision = VisionAgent()
        self.weather = WeatherAgent()
        self.market = MarketIntelligenceAgent()
        self.budget = BudgetPlanningAgent()
        self.crop_knowledge = CropKnowledgeAgent()
        self.government_schemes = GovernmentSchemeAgent()
        self.task_planning = TaskPlanningAgent()
        self.crop_prediction = CropPredictionAgent()
        self.disease_prediction = DiseasePredictionAgent()
        self.waste_to_wealth = WasteToWealthAgent()
        self.soil_nutrient = SoilNutrientAgent()
        self.entomologist = EntomologistAgent()
        self.irrigation = IrrigationAgent()
        self.memory = MemoryAgent()
        self.storage = StorageService()
        self.evaluator = AgentEvaluator.get_shared()

        self._agent_map = {
            "vision": self.vision,
            "weather": self.weather,
            "market": self.market,
            "budget": self.budget,
            "crop_knowledge": self.crop_knowledge,
            "government_schemes": self.government_schemes,
            "task_planning": self.task_planning,
            "crop_prediction": self.crop_prediction,
            "disease_prediction": self.disease_prediction,
            "waste_to_wealth": self.waste_to_wealth,
            "soil_nutrient": self.soil_nutrient,
            "entomologist": self.entomologist,
            "irrigation": self.irrigation,
        }

        self.execution_mode = "auto"
        self.routing_strategy: str = "auto"

    async def process_stream(self, context: Dict[str, Any]):
        pipeline_start = time.time()
        request_id = context.get("request_id", str(uuid.uuid4())[:8])
        trace_id = self.evaluator.start_trace(context, request_id)
        user_id = context.get("user_id", "default_user")
        ts = lambda: datetime.now(timezone.utc).isoformat()

        agent_results: Dict[str, AgentResult] = {}
        succeeded_agents: List[str] = []
        failed_agents: List[str] = []
        total_agent_tokens = 0
        agent_events: List[Dict] = []
        fusion = None
        tier2_tasks: List[asyncio.Task] = []

        try:
            # ── Check cache for identical queries ─────────────────────────
            cache_key = _cache_key(context)
            cached = _get_cached(cache_key)
            if cached and not context.get("image_bytes") and not context.get("image_base64"):
                self.logger.info(f"Cache HIT for key {cache_key[:8]} — skipping pipeline")
                cached["_from_cache"] = True
                yield cached
                return

            # ── Step 1: Upload image if present ────────────────────────
            image_bytes = context.get("image_bytes")
            if image_bytes and self.storage.available:
                yield self._event("🖼 Uploading image to secure storage", ts(), status="running")
                upload_result = self.storage.upload_image(
                    image_bytes, f"leaf_{int(time.time())}.jpg", user_id
                )
                if upload_result.get("uploaded"):
                    context["image_storage_path"] = upload_result["path"]
                    context["image_signed_url"] = upload_result.get("signed_url", "")
                    yield self._event(
                        "🖼 Image stored securely", ts(),
                        status="completed",
                        tools_used=["Supabase Storage"],
                        metadata={"size_mb": upload_result.get("size_mb", 0)},
                    )
                else:
                    yield self._event(
                        f"🖼 Image upload failed: {upload_result.get('error', 'Unknown')}",
                        ts(), status="failed",
                    )

            # ── Step 2: Load memory (concurrent with routing) ──────────
            mem_start = time.time()
            memory_future = asyncio.create_task(
                asyncio.to_thread(self.memory.process, {"user_id": user_id, "memory_action": "retrieve"})
            )

            crop_present = bool(context.get("crop") and context.get("crop") != "Unknown" and str(context.get("crop")).strip() != "")
            loc_present = bool(context.get("location") and context.get("location") != "Unknown" and str(context.get("location")).strip() != "")

            if not (crop_present and loc_present):
                yield self._event("📂 Loading Farm Memory", ts(), status="running")
                farm_memory = await memory_future
                mem_latency = (time.time() - mem_start) * 1000
                if farm_memory and farm_memory.get("status") not in ("skipped", "no_memory_found"):
                    context["farm_memory"] = farm_memory
                    yield self._event("📂 Farm memory loaded", ts(), status="completed", latency_ms=mem_latency, data_sources=["Firestore"])
                else:
                    yield self._event("📂 No previous memory found", ts(), status="completed", latency_ms=mem_latency)
            else:
                farm_memory = None
                yield self._event("📂 Farm profile active", ts(), status="completed", latency_ms=0)

            # ── Step 3: Context inference ──────────────────────────────
            yield self._event("⚡ Checking context details for missing fields...", ts(), status="running")
            self._infer_missing_fields(context, farm_memory)
            yield self._event(
                f"⚡ Context resolved: Location={context.get('location')}, Crop={context.get('crop')}, Budget=₹{context.get('budget')}",
                ts(), status="completed",
            )

            # ── Step 4: Deterministic routing + agent selection ────────
            tier1_agents, tier2_agents, routing_tag = self._resolve_routing(context)
            self.routing_strategy = routing_tag

            yield self._event(
                f"🎯 Routing: {routing_tag.replace('_', ' ').title()} | Tier-1: {', '.join(tier1_agents)}"
                + (f" | Tier-2: {', '.join(tier2_agents)}" if tier2_agents else ""),
                ts(), status="completed",
                metadata={"routing": routing_tag, "tier1": tier1_agents, "tier2": tier2_agents},
            )

            # ── Step 5: Execute Tier 1 concurrently ───────────────────
            tier1_timeout = 25.0
            yield self._event(
                f"⚡ Executing Tier-1 core ({len(tier1_agents)} agents in parallel)",
                ts(), status="running",
            )

            tier1_start = time.time()
            async def _staggered_exec(idx: int, name: str):
                if idx > 0:
                    await asyncio.sleep(idx * 0.04)
                return await self._execute_agent_with_retry(name, context, trace_id)

            tier1_tasks = {
                name: asyncio.create_task(_staggered_exec(i, name))
                for i, name in enumerate(tier1_agents)
            }
            try:
                done_t1, pending_t1 = await asyncio.wait(
                    tier1_tasks.values(),
                    timeout=tier1_timeout,
                )
            except Exception as wait_err:
                self.logger.warning(f"Tier-1 wait error: {wait_err}")
                done_t1, pending_t1 = set(), set(tier1_tasks.values())

            for task in pending_t1:
                task.cancel()

            tier1_latency = (time.time() - tier1_start) * 1000

            for name in tier1_agents:
                task = tier1_tasks[name]
                if task in done_t1 and not task.cancelled():
                    try:
                        res = task.result()
                        if isinstance(res, Exception):
                            ar = AgentResult(
                                status=AgentStatus.FAILED,
                                agent_name=name,
                                execution_time_ms=0,
                                error=str(res),
                            )
                        else:
                            ar = res
                    except Exception as res_err:
                        ar = AgentResult(
                            status=AgentStatus.FAILED,
                            agent_name=name,
                            execution_time_ms=0,
                            error=str(res_err),
                        )
                else:
                    ar = AgentResult(
                        status=AgentStatus.TIMEOUT,
                        agent_name=name,
                        execution_time_ms=round(tier1_timeout * 1000, 1),
                        error=f"Agent '{name}' timed out after {tier1_timeout}s",
                    )

                agent_results[name] = ar
                agent_events.append({
                    "agent": name,
                    "status": ar.status.value,
                    "success": ar.succeeded,
                    "latency_ms": ar.execution_time_ms,
                    "latency_sec": round(ar.execution_time_ms / 1000, 3),
                    "tokens": ar.tokens_used,
                    "confidence": ar.confidence,
                    "error": ar.error,
                    "tier": 1,
                })

                if ar.succeeded:
                    succeeded_agents.append(name)
                    total_agent_tokens += ar.tokens_used
                    context[f"{name}_result"] = ar.data
                    yield self._event(
                        f"{EMOJI_MAP.get(name, '⚙️')} {name.replace('_', ' ').title()} ✓",
                        ts(), agent=name, latency_ms=ar.execution_time_ms,
                        status="completed",
                        metadata={"tokens": ar.tokens_used, "confidence": round(ar.confidence, 1), "tier": 1},
                    )
                else:
                    failed_agents.append(name)
                    error_short = (ar.error or "Unknown error")[:120]
                    yield self._event(
                        f"{EMOJI_MAP.get(name, '⚙️')} {name.replace('_', ' ').title()} ✗ — {error_short}",
                        ts(), agent=name, latency_ms=ar.execution_time_ms,
                        status="failed", error=ar.error, metadata={"tier": 1},
                    )

            # ── Step 6: Instant Fusion after Tier 1 ───────────────────
            if succeeded_agents:
                yield self._event("🧠 Fusing Tier-1 decisions", ts(), status="running")
                fusion_start = time.time()
                fusion = await self._fuse_decisions(
                    agent_results, context,
                    list(agent_results.keys()),
                    succeeded_agents, failed_agents,
                )
                fusion = self._apply_guardrails(fusion, context, agent_results)
                fusion_latency = (time.time() - fusion_start) * 1000
                yield self._event(
                    "🧠 Fusion complete", ts(),
                    status="completed", latency_ms=fusion_latency,
                    metadata={"ttft_ms": round(tier1_latency + fusion_latency, 1)},
                )

                # Emit early fusion result (TTFT event for frontend)
                yield {
                    "type": "pre_result",
                    "message": "Tier-1 fusion ready — Tier-2 enrichment streaming in",
                    "fusion": fusion.model_dump(),
                    "ttft_ms": round(tier1_latency + fusion_latency, 1),
                    "timestamp": ts(),
                }

                # Reflection on Tier-1 fusion (only if low confidence < 60% or critical risk)
                critical_failed = [a for a in failed_agents if a not in NON_CRITICAL_AGENTS]
                should_reflect = (
                    not critical_failed 
                    and not fusion.is_degraded 
                    and (fusion.confidence_score < 60.0 or fusion.risk_level == "Critical")
                )
                if should_reflect:
                    yield self._event("🔍 Running Reflection", ts(), status="running")
                    reflect_start = time.time()
                    fusion = await self._reflect_on_decision(fusion, context)
                    reflect_latency = (time.time() - reflect_start) * 1000
                    yield self._event(
                        "🔍 Reflection complete", ts(),
                        status="completed", latency_ms=reflect_latency,
                    )
            else:
                yield self._event(
                    "🧠 Fusion skipped — all Tier-1 agents failed", ts(),
                    status="failed", error="No Tier-1 agent data",
                )
                fusion = FusionResult(
                    summary="Unable to generate agricultural advice. All core agents failed.",
                    recommended_actions=[],
                    confidence_score=0,
                    confidence_label="Low",
                    risk_level="Unknown",
                    is_degraded=True,
                    failed_agents=failed_agents,
                    degradation_reasons=[
                        f"Agent '{a}' failed: {agent_results[a].error}"
                        for a in failed_agents if a in agent_results
                    ],
                )

            # ── Step 7: Tier 2 enrichment (Timeline, Schemes, Waste, Market) ─
            if tier2_agents:
                yield self._event(
                    f"🚀 Launching Tier-2 enrichment: {', '.join(tier2_agents)}",
                    ts(), status="running",
                    metadata={"agents": tier2_agents, "tier": 2},
                )
                t2_tasks = {
                    name: asyncio.create_task(self._execute_agent_with_retry(name, context, trace_id))
                    for name in tier2_agents
                }
                try:
                    done_t2, pending_t2 = await asyncio.wait(t2_tasks.values(), timeout=12.0)
                except Exception as t2_err:
                    self.logger.warning(f"Tier-2 wait error: {t2_err}")
                    done_t2, pending_t2 = set(), set(t2_tasks.values())

                for task in pending_t2:
                    task.cancel()

                for name in tier2_agents:
                    task = t2_tasks[name]
                    if task in done_t2 and not task.cancelled():
                        try:
                            res = task.result()
                            if isinstance(res, Exception):
                                ar = AgentResult(
                                    status=AgentStatus.FAILED,
                                    agent_name=name,
                                    execution_time_ms=0,
                                    error=str(res),
                                )
                            else:
                                ar = res
                        except Exception as res_err:
                            ar = AgentResult(
                                status=AgentStatus.FAILED,
                                agent_name=name,
                                execution_time_ms=0,
                                error=str(res_err),
                            )
                    else:
                        ar = AgentResult(
                            status=AgentStatus.TIMEOUT,
                            agent_name=name,
                            execution_time_ms=12000,
                            error=f"Tier-2 agent '{name}' timed out",
                        )

                    agent_results[name] = ar
                    agent_events.append({
                        "agent": name,
                        "status": ar.status.value,
                        "success": ar.succeeded,
                        "latency_ms": ar.execution_time_ms,
                        "latency_sec": round(ar.execution_time_ms / 1000, 3),
                        "tokens": ar.tokens_used,
                        "confidence": ar.confidence,
                        "error": ar.error,
                        "tier": 2,
                    })
                    if ar.succeeded:
                        succeeded_agents.append(name)
                        total_agent_tokens += ar.tokens_used
                        context[f"{name}_result"] = ar.data
                        yield self._event(
                            f"{EMOJI_MAP.get(name, '⚙️')} {name.replace('_', ' ').title()} ✓",
                            ts(), agent=name, latency_ms=ar.execution_time_ms,
                            status="completed",
                            metadata={"tokens": ar.tokens_used, "confidence": round(ar.confidence, 1), "tier": 2},
                        )
                    else:
                        failed_agents.append(name)
                        error_short = (ar.error or "Unknown error")[:120]
                        yield self._event(
                            f"{EMOJI_MAP.get(name, '⚙️')} {name.replace('_', ' ').title()} ✗ — {error_short}",
                            ts(), agent=name, latency_ms=ar.execution_time_ms,
                            status="failed", error=ar.error, metadata={"tier": 2},
                        )

            if farm_memory is None and 'memory_future' in locals():
                try:
                    farm_memory = await asyncio.wait_for(asyncio.shield(memory_future), timeout=1.0)
                    if farm_memory and farm_memory.get("status") not in ("skipped", "no_memory_found"):
                        context["farm_memory"] = farm_memory
                except Exception:
                    pass

            # ── Step 8: Firestore persistence (batched, non-blocking) ─
            step9_start = time.time()
            pipeline_latency_ms = (step9_start - pipeline_start) * 1000
            fusion_data_local = fusion.model_dump() if fusion else {}
            conf_label_local = (fusion.confidence_label or FusionResult.confidence_to_label(fusion.confidence_score)) if fusion else "Low"

            yield self._event("💾 Saving to Firestore", ts(), status="running")
            try:
                persist_payload = {
                    "user_id": user_id,
                    "memory_action": "batch_persist",
                }

                # Farm memory
                persist_payload["farm_memory"] = {
                    "last_query": context.get("text_query", ""),
                    "last_recommendation": fusion_data_local,
                    "crop": context.get("crop", ""),
                    "location": context.get("location", ""),
                    "user_id": user_id,
                }

                # Conversation history
                persist_payload["conversation"] = {
                    "user_id": user_id,
                    "query": context.get("text_query", ""),
                    "recommendation": fusion_data_local,
                    "agents_used": list(succeeded_agents),
                    "agents_failed": failed_agents,
                    "confidence": conf_label_local,
                    "success_rate": round((len(succeeded_agents) / max(len(agent_results), 1)) * 100, 1) if agent_results else 0,
                }

                # Execution trace
                persist_payload["execution_trace"] = {
                    "user_id": user_id,
                    "trace_id": trace_id,
                    "request_id": request_id,
                    "total_latency_sec": round(pipeline_latency_ms / 1000, 3),
                    "total_latency_ms": round(pipeline_latency_ms, 1),
                    "total_tokens": total_agent_tokens,
                    "status": "Completed",
                    "confidence_score": conf_label_local,
                    "agents_invoked": agent_events,
                    "agents_succeeded": succeeded_agents,
                    "agents_failed": failed_agents,
                    "execution_mode": self.routing_strategy,
                    "routed_mode": context.get("routed_mode", "auto"),
                    "source": "orchestrator",
                }

                # Disease history (conditional)
                vision_result = agent_results.get("vision")
                if (vision_result and vision_result.succeeded
                        and vision_result.data.get("disease")
                        and vision_result.data["disease"] != "Healthy"):
                    persist_payload["disease"] = {
                        "user_id": user_id,
                        "disease": vision_result.data.get("disease", ""),
                        "crop": context.get("crop", ""),
                        "severity": vision_result.data.get("severity", ""),
                        "confidence": vision_result.data.get("confidence", 0),
                        "image_path": context.get("image_storage_path", ""),
                    }

                # Write to Firestore (via EventBus or direct)
                try:
                    if bus_is_initialized():
                        bus = get_bus()
                        await bus.emit({"type": "persist", "payload": persist_payload})
                    else:
                        await asyncio.to_thread(self.memory.process, persist_payload)
                    yield self._event("💾 Saved to Firestore", ts(), status="completed")
                except Exception as persist_err:
                    # Fallback: try direct write if EventBus fails
                    logger.warning(f"EventBus persist failed, trying direct: {persist_err}")
                    try:
                        await asyncio.to_thread(self.memory.process, persist_payload)
                        yield self._event("💾 Saved to Firestore (direct fallback)", ts(), status="completed")
                    except Exception as direct_err:
                        raise direct_err
            except Exception as e:
                yield self._event(
                    f"💾 Firestore batch save failed: {str(e)[:80]}", ts(),
                    status="failed", error=str(e),
                )

        except Exception as pipeline_error:
            self.logger.error(f"Pipeline error: {pipeline_error}\n{traceback.format_exc()}")
            err_str = str(pipeline_error).strip() or type(pipeline_error).__name__
            if fusion is None:
                fusion = FusionResult(
                    summary=f"Execution notice: {err_str[:200]}",
                    recommended_actions=["Please retry the simulation or verify your network connection."],
                    confidence_score=0,
                    confidence_label="Low",
                    risk_level="Unknown",
                    is_degraded=True,
                    failed_agents=failed_agents or [],
                    degradation_reasons=[f"Pipeline error: {err_str[:100]}"],
                )

        # ── Finalize: emit trace and final result ──────────────────────
        total_latency = (time.time() - pipeline_start) * 1000

        fusion_data = fusion.model_dump() if fusion else {}
        conf_label = (fusion.confidence_label or FusionResult.confidence_to_label(fusion.confidence_score)) if fusion else "Low"

        await asyncio.to_thread(
            self.evaluator.end_trace,
            trace_id,
            fusion_data,
            conf_label,
            summary=fusion.summary if fusion else "Pipeline encountered an error.",
            total_latency_ms=total_latency,
            total_tokens=total_agent_tokens,
        )

        final_status = "completed" if (fusion and not (fusion.is_degraded and len(succeeded_agents) == 0)) else "failed"
        if fusion and fusion.is_degraded and len(succeeded_agents) > 0:
            final_status = "completed"

        yield self._event(
            "✅ Recommendation Ready" if final_status == "completed" else "❌ Unable to Generate Recommendation",
            ts(), latency_ms=total_latency, status=final_status,
        )

        agent_results_raw = {}
        agent_results_meta = {}
        for name, ar in agent_results.items():
            agent_results_raw[name] = ar.data if ar.succeeded else {"error": ar.error}
            agent_results_meta[name] = {
                "status": ar.status.value,
                "execution_time_ms": round(ar.execution_time_ms, 1),
                "tokens_used": ar.tokens_used,
                "confidence": round(ar.confidence, 1),
                "model": ar.model,
                "error": ar.error,
                "tool_calls": ar.tool_calls,
            }

        total_succeeded = len(succeeded_agents)
        total_executed = len(agent_results) if agent_results else 1
        success_rate = round((total_succeeded / total_executed) * 100, 1) if total_executed else 0

        # Cache the result for identical queries
        if cache_key:
            _set_cache(cache_key, {
                "type": "result",
                "data": {**fusion_data, "confidence": conf_label},
                "_from_cache": False,
            })

        yield {
            "type": "result",
            "data": {
                **fusion_data,
                "confidence": conf_label,
            },
            "agent_results": agent_results_raw,
            "agent_results_meta": agent_results_meta,
            "execution_metadata": {
                "request_id": request_id,
                "total_latency_ms": round(total_latency, 1),
                "agents_executed": list(agent_results.keys()),
                "agents_succeeded": succeeded_agents,
                "agents_failed": failed_agents,
                "total_tokens": total_agent_tokens,
                "trace_id": trace_id,
                "timestamp": ts(),
                "is_degraded": fusion.is_degraded if fusion else True,
                "routed_mode": self.routing_strategy,
                "success_rate": success_rate,
                "agent_events": agent_events,
            },
            "execution_report": ExecutionReport(
                request_id=request_id,
                trace_id=trace_id,
                query=context.get("text_query", ""),
                location=context.get("location", ""),
                crop=context.get("crop", ""),
                budget=context.get("budget", 0),
                execution_mode=self.routing_strategy,
                total_latency_ms=round(total_latency, 1),
                total_tokens=total_agent_tokens,
                agents_executed=list(agent_results.keys()),
                agents_succeeded=succeeded_agents,
                agents_failed=failed_agents,
                fusion_result=fusion_data,
                agent_results=agent_results_raw,
                agent_metadata=agent_results_meta,
                performance_metrics={
                    "total_agents": len(agent_results) if agent_results else 0,
                    "succeeded": len(succeeded_agents),
                    "failed": len(failed_agents),
                    "success_rate": success_rate,
                    "total_latency_ms": round(total_latency, 1),
                    "total_tokens": total_agent_tokens,
                    "confidence": conf_label,
                },
                execution_log=agent_events,
                is_degraded=fusion.is_degraded if fusion else True,
                status=final_status,
            ).model_dump(),
        }

    # ── Deterministic Routing (Req #2) ──────────────────────────────────

    def _resolve_routing(self, context: Dict[str, Any]) -> Tuple[List[str], List[str], str]:
        """
        Deterministic routing with short-circuits:
        - Vision bypass: image present → vision + disease_prediction
        - Keyword shortcut: market keywords → market_intelligence
        - Fall through to mode-based routing otherwise.
        Returns (tier1_agents, tier2_agents, routing_tag).
        """
        has_image = bool(context.get("image_bytes") or context.get("image_base64"))
        query_lower = context.get("text_query", "").lower()
        execution_mode = context.get("execution_mode", "auto")

        # ── Vision Bypass (short-circuit) ──────────────────────────────
        if has_image:
            context["routed_mode"] = "vision_bypass"
            self.execution_mode = "vision_bypass"
            return (["vision", "disease_prediction"], list(TIER2_ENRICHMENT), "vision_bypass")

        # ── Keyword Shortcut (market) ──────────────────────────────────
        if any(kw in query_lower for kw in MARKET_KEYWORDS):
            context["routed_mode"] = "market_shortcut"
            self.execution_mode = "market_shortcut"
            return (["market"], list(TIER2_ENRICHMENT), "market_shortcut")

        # ── Mode-based routing ─────────────────────────────────────────
        self.execution_mode = execution_mode

        if execution_mode == "fast_track":
            context["routed_mode"] = "fast_track"
            return (["disease_prediction"], list(TIER2_ENRICHMENT), "fast_track")

        if execution_mode == "specific":
            specific = context.get("specific_agent")
            if specific in self._agent_map:
                context["routed_mode"] = f"specific_{specific}"
                return ([specific], list(TIER2_ENRICHMENT), f"specific_{specific}")
            context["routed_mode"] = "specific_weather"
            return (["weather"], list(TIER2_ENRICHMENT), "specific_weather")

        if execution_mode == "all":
            all_agents = list(self._agent_map.keys())
            tier1 = [a for a in all_agents if a in TIER1_CORE]
            tier2 = [a for a in all_agents if a in TIER2_ENRICHMENT]
            rest = [a for a in all_agents if a not in TIER1_CORE and a not in TIER2_ENRICHMENT]
            context["routed_mode"] = "all"
            return (tier1 + rest, tier2, "all")

        if execution_mode == "ensemble":
            # Ensemble: core + disease_prediction + market (matches original frontend mode)
            context["routed_mode"] = "ensemble"
            tier1 = list(TIER1_CORE) + ["disease_prediction", "market"]
            return (tier1, list(TIER2_ENRICHMENT), "ensemble")

        # Auto: Tier 1 core + Fast Deterministic / Specialist Routing + Tier 2
        context["routed_mode"] = "auto"
        additional = self._resolve_specialists_fast(context)
        query_lower = (context.get("text_query") or "").lower()
        if any(kw in query_lower for kw in MARKET_KEYWORDS):
            tier1 = list(TIER1_CORE) + ["market"] + additional
            tier2 = list(TIER2_ENRICHMENT)
        else:
            tier1 = list(TIER1_CORE) + additional
            tier2 = list(TIER2_ENRICHMENT) + ["market"]
        return (tier1, tier2, "auto")

    def _resolve_specialists_fast(self, context: Dict[str, Any]) -> List[str]:
        """
        Ultra-fast (<0.1ms) deterministic specialist resolution using agricultural keyword taxonomy.
        Eliminates blocking LLM roundtrips.
        """
        query_lower = (context.get("text_query") or "").lower()
        specialists: Set[str] = set()

        pest_keywords = {
            "pest", "pests", "bug", "bugs", "worm", "worms", "borer", "aphid", "caterpillar", "insect", "insects", "beetle",
            "mite", "mites", "whitefly", "thrips", "larvae", "grub", "infest", "infestation", "locust", "leafhopper",
            "stem borer", "bollworm", "armyworm", "cutworm"
        }
        if any(kw in query_lower for kw in pest_keywords):
            specialists.add("entomologist")

        water_keywords = {
            "water", "irrigation", "drip", "sprinkler", "moisture", "dry", "watering",
            "drought", "wilting", "waterlogging", "flood", "subsurface",
            "rain", "rainfall", "rainy", "shower", "monsoon", "humid", "humidity"
        }
        if any(kw in query_lower for kw in water_keywords):
            specialists.add("irrigation")

        disease_keywords = {
            "disease", "blight", "rot", "mildew", "spot", "spots", "yellowing", "wilt", "fungus",
            "rust", "canker", "scab", "mosaic", "curl", "lesion", "spore", "fungal", "bacterial",
            "spray", "fungicide", "pesticide", "cure", "protect", "prevent", "damage", "attack",
            "infection", "leaf", "leaves"
        }
        if any(kw in query_lower for kw in disease_keywords):
            specialists.add("disease_prediction")

        crop_keywords = {
            "what to plant", "which crop", "recommend crop", "crop selection", "best crop",
            "next season", "alternate crop", "crop rotation", "profitable crop", "sow"
        }
        if any(kw in query_lower for kw in crop_keywords):
            specialists.add("crop_prediction")

        waste_keywords = {
            "waste", "stubble", "residue", "husk", "straw", "stalk", "bagasse", "biochar",
            "compost", "biogas"
        }
        if any(kw in query_lower for kw in waste_keywords):
            specialists.add("waste_to_wealth")

        if specialists:
            return [s for s in specialists if s in self._agent_map]

        if context.get("crop") and context.get("crop") != "Unknown":
            return ["disease_prediction"]

        return []

    def _resolve_with_llm(self, context: Dict[str, Any]) -> List[str]:
        """Fast fallback: returns disease_prediction for known crops without blocking."""
        if context.get("crop") and context.get("crop") != "Unknown":
            return ["disease_prediction"]
        return []

    # ── Tier 1 Execution (concurrent gather) ────────────────────────────

    async def _execute_tier1_agents(
        self,
        names: List[str],
        context: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, AgentResult]:
        coros = [self._execute_agent_with_retry(n, context, trace_id) for n in names]
        results = await asyncio.gather(*coros, return_exceptions=True)
        out = {}
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                out[name] = AgentResult(
                    status=AgentStatus.FAILED,
                    agent_name=name,
                    execution_time_ms=0,
                    error=str(result),
                )
            else:
                out[name] = result
        return out

    # ── Guardrails (Req #4) ────────────────────────────────────────────

    def _apply_guardrails(
        self,
        fusion: FusionResult,
        context: Dict[str, Any],
        agent_results: Dict[str, AgentResult],
    ) -> FusionResult:
        weather = agent_results.get("weather")
        budget_val = context.get("budget", 0)

        # Weather Override: rain > 60% within 48h → strip/delay chemical sprays
        if weather and weather.succeeded:
            weather_data = weather.data or {}
            rain_prob = weather_data.get("rain_probability", 0)
            if rain_prob > 60:
                filtered = []
                spray_warning = (
                    f"⚠️ Rain probability {rain_prob}% > 60% threshold. "
                    "Chemical spray delayed. Biological/cultural controls recommended."
                )
                for action in fusion.recommended_actions:
                    action_lower = action.lower()
                    if any(kw in action_lower for kw in ["spray", "pesticide", "fungicide", "chemical", "herbicide"]):
                        filtered.append(f"🕒 [DELAYED] {action} — Apply after rain passes")
                    else:
                        filtered.append(action)
                if filtered:
                    fusion.recommended_actions = [spray_warning] + filtered
                else:
                    fusion.recommended_actions = [spray_warning] + fusion.recommended_actions
                fusion.weather_consideration = (
                    f"Rain probability {rain_prob}% within 48h. "
                    "Chemical applications postponed. Foliar treatments deferred."
                )

        # Budget Ceiling: if total treatment cost > budget → DEGRADED_BUDGET_WARN
        if budget_val and budget_val > 0:
            total_cost = 0
            for ar in agent_results.values():
                if ar.succeeded and ar.data:
                    cost = ar.data.get("total_cost") or ar.data.get("estimated_cost") or 0
                    total_cost += cost
            if total_cost > budget_val * 1.1:
                fusion.is_degraded = True
                fusion.degradation_reasons.append(
                    f"DEGRADED_BUDGET_WARN: Total estimated cost ₹{total_cost:.0f} exceeds budget ₹{budget_val:.0f} by ₹{total_cost - budget_val:.0f}"
                )
                fusion.budget_assessment = (
                    f"Cost ₹{total_cost:.0f} exceeds budget ₹{budget_val:.0f}. "
                    "Low-cost biological/cultural alternatives injected."
                )
                # Inject low-cost alternatives at the top of actions
                low_cost_actions = [
                    f"🌱 [BUDGET-FRIENDLY] Use neem oil spray (₹250) instead of chemical pesticide",
                    f"🌱 [BUDGET-FRIENDLY] Apply compost tea for soil health (₹100)",
                    f"🌱 [BUDGET-FRIENDLY] Practice crop rotation for natural pest control",
                ]
                fusion.recommended_actions = low_cost_actions + fusion.recommended_actions

        return fusion

    # ── Agent Execution with Retry (async path) ─────────────────────────

    async def _execute_agent_with_retry(
        self, name: str, context: Dict[str, Any], trace_id: str
    ) -> AgentResult:
        agent = self._agent_map.get(name)
        if not agent:
            return AgentResult(
                status=AgentStatus.FAILED,
                agent_name=name,
                error=f"Unknown agent: {name}",
            )

        for attempt in range(2):
            try:
                result = await asyncio.wait_for(
                    agent.async_execute(context),
                    timeout=AGENT_TIMEOUT_SECONDS,
                )
                self.evaluator.log_agent_execution(
                    trace_id, name, result.succeeded,
                    result.execution_time_ms / 1000,
                    result.data if result.succeeded else {"error": result.error},
                )
                if result.succeeded:
                    return result
                if attempt == 0 and self._is_transient_error(result.error):
                    self.logger.info(f"Retrying agent {name} after transient failure...")
                    await asyncio.sleep(0.3)
                    continue
                return result
            except asyncio.TimeoutError:
                self.evaluator.log_agent_execution(trace_id, name, False, AGENT_TIMEOUT_SECONDS, {"error": "timeout"})
                return AgentResult(
                    status=AgentStatus.TIMEOUT,
                    agent_name=name,
                    execution_time_ms=AGENT_TIMEOUT_SECONDS * 1000,
                    error=f"Agent timed out after {AGENT_TIMEOUT_SECONDS}s",
                )
            except Exception as e:
                self.evaluator.log_agent_execution(trace_id, name, False, 0, {"error": str(e)})
                if attempt == 0 and self._is_transient_error(str(e)):
                    self.logger.info(f"Agent {name} transient exception, retrying: {e}")
                    await asyncio.sleep(0.3)
                    continue
                return AgentResult(
                    status=AgentStatus.FAILED,
                    agent_name=name,
                    error=str(e),
                )
        return AgentResult(
            status=AgentStatus.FAILED,
            agent_name=name,
            error="Max retry attempts exhausted",
        )

    @staticmethod
    def _is_transient_error(error: Optional[str]) -> bool:
        if not error:
            return False
        error_lower = error.lower()
        transient_keywords = [
            "rate limit", "429", "timeout", "quota", "resource exhausted",
            "unavailable", "service busy", "retry", "connection",
            "temporarily", "overloaded", "too many requests",
        ]
        return any(kw in error_lower for kw in transient_keywords)

    # ── Context Inference ──────────────────────────────────────────────

    def _infer_missing_fields(self, context: Dict[str, Any], farm_memory: Optional[Dict[str, Any]] = None):
        query_text = context.get("text_query", "").lower()

        loc_val = context.get("location")
        if not loc_val or loc_val == "Unknown" or str(loc_val).strip() == "":
            if farm_memory and farm_memory.get("location"):
                context["location"] = farm_memory["location"]
            else:
                locations = ["bangalore", "coimbatore", "pune", "chennai", "hyderabad", "kolkata", "lucknow", "delhi", "mumbai"]
                found_loc = next((loc.title() for loc in locations if loc in query_text), None)
                context["location"] = found_loc or "Bangalore"

        crop_val = context.get("crop")
        if not crop_val or crop_val == "Unknown" or str(crop_val).strip() == "":
            if farm_memory and farm_memory.get("crop"):
                context["crop"] = farm_memory["crop"]
            else:
                crops = ["tomato", "rice", "corn", "spinach", "lettuce", "sugarcane", "cotton", "wheat", "potato"]
                found_crop = next((c.title() for c in crops if c in query_text), None)
                context["crop"] = found_crop or "Tomato"

        budg_val = context.get("budget")
        if budg_val is None or budg_val == 0:
            if farm_memory and farm_memory.get("budget"):
                context["budget"] = float(farm_memory["budget"])
            else:
                import re
                budget_match = re.search(r'(?:budget|rs\.?|₹)\s*(\d+)', query_text)
                if budget_match:
                    context["budget"] = float(budget_match.group(1))
                else:
                    context["budget"] = 5000.0

    # ── Decision Fusion (async) ────────────────────────────────────────

    async def _fuse_decisions(
        self,
        agent_results: Dict[str, AgentResult],
        context: Dict[str, Any],
        agents_used: List[str],
        succeeded: List[str],
        failed: List[str],
    ) -> FusionResult:
        concise_data = {}
        for name, ar in agent_results.items():
            if ar.succeeded and ar.data:
                d = ar.data
                if isinstance(d, dict):
                    entry = {}
                    for key in [
                        "disease", "confidence", "weather_risk", "spray_safe", "rain_probability",
                        "temperature", "humidity", "irrigation_needed", "estimated_total_cost",
                        "budget_compliant", "best_value_option", "cheapest_option",
                        "top_deficiency", "soil_health_score", "predicted_diseases", "reasoning",
                        "summary", "immediate_action", "treatment"
                    ]:
                        if key in d:
                            val = d[key]
                            if isinstance(val, dict):
                                entry[key] = {k: v for k, v in val.items() if k in ["name", "cost_estimate_inr", "disease_name", "action"]}
                            elif isinstance(val, list):
                                entry[key] = val[:2]
                            else:
                                entry[key] = val
                    concise_data[name] = entry if entry else {k: str(v)[:120] for k, v in list(d.items())[:4]}
                else:
                    concise_data[name] = str(d)[:200]

        if succeeded:
            avg_confidence = sum(
                agent_results[a].confidence for a in succeeded
            ) / len(succeeded)
            success_ratio = len(succeeded) / len(agents_used) if agents_used else 0
            computed_confidence = int(avg_confidence * success_ratio)
        else:
            computed_confidence = 0

        is_degraded = len(failed) > 0

        if not gemini_service.is_available():
            return FusionResult(
                summary="Unable to generate recommendation — Gemini API unavailable.",
                confidence_score=0,
                confidence_label="Low",
                is_degraded=True,
                failed_agents=failed,
                agents_used=succeeded,
                degradation_reasons=["Gemini API authentication failed"],
            )

        prompt = (
            f"Farmer request: {context.get('text_query', '')}\n"
            f"Budget: ₹{context.get('budget', 'Not specified')} | "
            f"Location: {context.get('location', 'Unknown')} | "
            f"Crop: {context.get('crop', 'Unknown')}\n"
            f"Agent Findings: {concise_data}\n"
            f"Failed agents: {failed}\n"
            "Merge evidence into one coherent recommendation. "
            "Resolve conflicts (rain vs spray → delay). Respect budget. Prioritize safety. "
            "Rank by urgency. List evidence sources. Suggest alternatives if confidence not High. "
            "Note limitations from failed agents. Return ONLY valid JSON."
        )
        try:
            response = await self.async_call_llm(prompt, schema=LLMFusionSchema)
            clean_text = gemini_service.extract_clean_json(response.text)
            llm_result = LLMFusionSchema.model_validate_json(clean_text)

            fusion = FusionResult(
                summary=llm_result.summary,
                recommended_actions=llm_result.recommended_actions,
                confidence_score=computed_confidence,
                confidence_label=FusionResult.confidence_to_label(computed_confidence),
                risk_level=llm_result.risk_level,
                budget_assessment=llm_result.budget_assessment,
                weather_consideration=llm_result.weather_consideration,
                evidence_sources=llm_result.evidence_sources,
                agents_used=succeeded if succeeded else agents_used,
                tools_executed=llm_result.tools_executed,
                alternatives_considered=llm_result.alternatives_considered,
                government_schemes=llm_result.government_schemes,
                follow_up_actions=llm_result.follow_up_actions,
                is_degraded=is_degraded,
                failed_agents=failed,
                degradation_reasons=[
                    f"Agent '{a}' failed: {agent_results[a].error}"
                    for a in failed if a in agent_results
                ],
            )
            return fusion

        except Exception as e:
            self.logger.error(f"Fusion failed: {e}")
            self.logger.info("Falling back to local (non-LLM) decision fusion")
            return self._local_fusion(concise_data, agent_results, context, succeeded, failed, is_degraded)

    def _local_fusion(
        self,
        successful_data: Dict[str, Any],
        agent_results: Dict[str, AgentResult],
        context: Dict[str, Any],
        succeeded: List[str],
        failed: List[str],
        is_degraded: bool,
    ) -> FusionResult:
        """Local (non-LLM) fusion fallback when Gemini is exhausted."""
        summary_parts = []
        actions: List[str] = []
        evidence: List[str] = []
        reasons = []

        for name in succeeded:
            ar = agent_results.get(name)
            if not ar or not ar.data:
                continue
            data = ar.data
            evidence.append(f"{name}: {str(data)[:120]}")
            if isinstance(data, dict):
                if data.get("summary"):
                    summary_parts.append(f"[{name}] {data['summary']}")
                if data.get("disease"):
                    sev = data.get("severity", "Unknown")
                    summary_parts.append(f"[Disease] {data['disease']} ({sev})")
                    actions.append(f"Treat {data['disease']}: {data.get('treatment', 'Consult specialist')}")
                if data.get("rain_probability") is not None:
                    rp = data["rain_probability"]
                    if rp > 60:
                        actions.append(f"Delay chemical spray — rain probability {rp}%")
                if data.get("market_price"):
                    summary_parts.append(f"[Market] ₹{data['market_price']}/{data.get('unit', 'kg')}")
                if data.get("recommended_actions"):
                    for a in data["recommended_actions"]:
                        if isinstance(a, str):
                            actions.append(a)
                if data.get("total_cost"):
                    summary_parts.append(f"[Budget] Estimated cost: ₹{data['total_cost']}")
                if data.get("soil_health"):
                    summary_parts.append(f"[Soil] {data['soil_health']}")
                if data.get("irrigation_schedule"):
                    actions.append(f"Irrigation: {data['irrigation_schedule']}")
                if data.get("pest_risk"):
                    summary_parts.append(f"[Pests] {data['pest_risk']}")

        summary = ". ".join(summary_parts) if summary_parts else "Agricultural analysis complete."
        if failed:
            summary += f" | Agents unavailable: {', '.join(failed)}"

        if not actions:
            actions = ["Monitor crop health", "Continue regular irrigation", "Apply fertilizer as scheduled"]

        budget_val = context.get("budget", 0)
        if budget_val:
            actions.append(f"Stay within budget of ₹{budget_val}")

        for a in failed:
            ar = agent_results.get(a)
            if ar and ar.error:
                reasons.append(f"Agent '{a}' failed: {ar.error[:100]}")

        computed_confidence = int(
            (sum(agent_results[a].confidence for a in succeeded) / len(succeeded) if succeeded else 0)
            * (len(succeeded) / max(len(agent_results), 1))
        )

        return FusionResult(
            summary=summary,
            recommended_actions=list(dict.fromkeys(actions)),
            confidence_score=computed_confidence,
            confidence_label=FusionResult.confidence_to_label(computed_confidence),
            risk_level="Medium",
            budget_assessment=f"Budget ₹{budget_val}" if budget_val else None,
            evidence_sources=evidence,
            agents_used=succeeded if succeeded else [],
            is_degraded=True,
            failed_agents=failed,
            degradation_reasons=reasons or ["Fusion used local fallback (Gemini unavailable)"],
        )

    # ── Reflection (async) ─────────────────────────────────────────────

    async def _reflect_on_decision(self, fusion: FusionResult, context: Dict[str, Any]) -> FusionResult:
        prompt = (
            f"Farmer Request: {context.get('text_query', '')} | Budget: ₹{context.get('budget', 'Unknown')} | Location: {context.get('location', 'Unknown')}\n"
            f"Proposed Summary: {fusion.summary}\n"
            f"Actions: {fusion.recommended_actions[:4]}\n\n"
            "Quick sanity check: 1) Budget compliance 2) Spray safety vs rain 3) Any severe contradictions. "
            "Return ONLY valid JSON."
        )
        try:
            response = await self.async_call_llm(prompt, schema=LLMFusionSchema)
            clean_text = gemini_service.extract_clean_json(response.text)
            revised_llm = LLMFusionSchema.model_validate_json(clean_text)

            fusion.summary = revised_llm.summary
            fusion.recommended_actions = revised_llm.recommended_actions
            fusion.risk_level = revised_llm.risk_level
            fusion.budget_assessment = revised_llm.budget_assessment
            fusion.weather_consideration = revised_llm.weather_consideration
            fusion.evidence_sources = revised_llm.evidence_sources
            fusion.tools_executed = revised_llm.tools_executed
            fusion.alternatives_considered = revised_llm.alternatives_considered
            fusion.government_schemes = revised_llm.government_schemes
            fusion.follow_up_actions = revised_llm.follow_up_actions

            self.logger.info("Reflection completed successfully.")
            return fusion

        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            return fusion

    @staticmethod
    def _event(message: str, timestamp: str, **kwargs) -> dict:
        event = {"type": "event", "message": message, "timestamp": timestamp}
        for key in ("agent", "status", "latency_ms", "tools_used",
                     "data_sources", "metadata", "error"):
            if key in kwargs:
                val = kwargs[key]
                if key == "latency_ms":
                    event[key] = round(val, 1)
                else:
                    event[key] = val
        return event
