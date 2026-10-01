"""
SEED AI — Agent Evaluator (Production v3 — User-Isolated)
Records and benchmarks agent execution metrics per authenticated user.
Sanitizes context to avoid storing image bytes.
Guarantees a report is always generated, even on partial/full failure.
Each user's traces are strictly scoped to their UID.
"""
import time
import json
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger("Evaluator")


class AgentEvaluator:
    """
    Records execution traces with strict per-user isolation.
    Never stores raw image bytes or large binary data.
    Guarantees report generation with fallback content.
    """

    _shared_instance: Optional['AgentEvaluator'] = None

    def __new__(cls, *args, **kwargs):
        """Enable singleton via cls.get_shared() without breaking direct construction."""
        return super().__new__(cls)

    def __init__(self, log_path: str = "logs/evaluation_report.json"):
        if not hasattr(self, '_initialized'):
            if os.environ.get("VERCEL"):
                log_path = "/tmp/evaluation_report.json"
            self.log_path = log_path
            self.trace_data: Dict[str, list] = {}
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._initialized = True

    @classmethod
    def get_shared(cls) -> 'AgentEvaluator':
        """Returns a shared singleton evaluator instance."""
        if cls._shared_instance is None:
            cls._shared_instance = cls()
        return cls._shared_instance

    def _get_user_traces(self, user_id: str) -> list:
        if user_id not in self.trace_data:
            self.trace_data[user_id] = []
        return self.trace_data[user_id]

    def start_trace(self, context: Dict[str, Any], request_id: str = "") -> str:
        trace_id = f"trace_{int(time.time())}_{request_id}"
        user_id = context.get("user_id", "anonymous")

        safe_context = {
            k: v for k, v in context.items()
            if k not in ("image_bytes", "image_base64", "farm_memory")
            and not isinstance(v, bytes)
        }

        user_traces = self._get_user_traces(user_id)
        user_traces.append({
            "trace_id": trace_id,
            "user_id": user_id,
            "request_id": request_id,
            "context": safe_context,
            "start_time": time.time(),
            "agents_invoked": [],
            "agents_succeeded": 0,
            "agents_failed": 0,
            "status": "Running",
        })
        return trace_id

    def log_agent_execution(
        self,
        trace_id: str,
        agent_name: str,
        success: bool,
        latency: float,
        data: Any = None,
    ):
        for user_id, user_traces in self.trace_data.items():
            for trace in user_traces:
                if trace["trace_id"] == trace_id:
                    trace["agents_invoked"].append({
                        "agent": agent_name,
                        "latency_sec": round(latency, 3),
                        "success": success,
                    })
                    if success:
                        trace["agents_succeeded"] += 1
                    else:
                        trace["agents_failed"] += 1
                    return

    def end_trace(
        self,
        trace_id: str,
        final_recommendation: Optional[Dict[str, Any]],
        confidence: str,
        summary: str = "",
        total_latency_ms: float = 0.0,
        total_tokens: int = 0,
    ):
        for user_id, user_traces in self.trace_data.items():
            for trace in user_traces:
                if trace["trace_id"] == trace_id:
                    trace["end_time"] = time.time()
                    trace["total_latency_sec"] = round(
                        trace["end_time"] - trace["start_time"], 3
                    )
                    trace["status"] = "Completed"
                    trace["confidence_score"] = confidence
                    trace["total_latency_ms"] = round(total_latency_ms, 1)
                    trace["total_tokens"] = total_tokens

                    if final_recommendation:
                        trace["final_recommendation"] = final_recommendation
                    else:
                        trace["final_recommendation"] = {
                            "summary": summary or "Execution completed with partial data. Review individual agent outputs for details.",
                            "confidence_score": 0,
                            "confidence_label": "Low",
                            "is_degraded": True,
                        }

                    report = {
                        "trace": trace,
                        "execution_summary": self._generate_summary(trace),
                        "report_generated_at": datetime.now(timezone.utc).isoformat(),
                    }

                    for attempt in range(3):
                        try:
                            with open(self.log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(report) + "\n")
                            logger.info(
                                f"Trace {trace_id} (user={user_id}) saved. "
                                f"Latency: {trace['total_latency_sec']:.2f}s, "
                                f"Succeeded: {trace['agents_succeeded']}, "
                                f"Failed: {trace['agents_failed']}"
                            )
                            break
                        except Exception as e:
                            logger.error(f"Failed to save trace (attempt {attempt + 1}): {e}")
                            if attempt < 2:
                                time.sleep(0.5)
                    return

    def _generate_summary(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        agents = trace.get("agents_invoked", [])
        succeeded = [a for a in agents if a.get("success")]
        failed = [a for a in agents if not a.get("success")]
        latencies = [a.get("latency_sec", 0) for a in agents]

        return {
            "request_id": trace.get("request_id", ""),
            "trace_id": trace.get("trace_id", ""),
            "user_id": trace.get("user_id", ""),
            "total_agents": len(agents),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "average_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "total_latency_sec": trace.get("total_latency_sec", 0),
            "confidence": trace.get("confidence_score", "Low"),
            "agent_names": [a.get("agent") for a in agents],
            "failed_agent_names": [a.get("agent") for a in failed],
        }

    def get_user_traces(self, user_id: str) -> List[Dict[str, Any]]:
        """Returns all trace data for a specific user with summaries."""
        user_traces = self._get_user_traces(user_id)
        return [
            {"trace": t, "execution_summary": self._generate_summary(t)}
            for t in user_traces
        ]

    def get_user_summary_stats(self, user_id: str) -> Dict[str, Any]:
        """Aggregate statistics across all traces for a specific user."""
        user_traces = self._get_user_traces(user_id)
        if not user_traces:
            return {
                "total_simulations": 0,
                "total_agents_executed": 0,
                "avg_latency_sec": 0,
                "total_tokens": 0,
                "success_rate": 0,
            }

        total_agents = sum(len(t.get("agents_invoked", [])) for t in user_traces)
        total_succeeded = sum(t.get("agents_succeeded", 0) for t in user_traces)
        total_latencies = [
            t.get("total_latency_sec", 0) for t in user_traces if t.get("total_latency_sec")
        ]
        total_tokens = sum(t.get("total_tokens", 0) for t in user_traces)

        return {
            "total_simulations": len(user_traces),
            "total_agents_executed": total_agents,
            "total_agents_succeeded": total_succeeded,
            "avg_latency_sec": round(
                sum(total_latencies) / len(total_latencies), 2
            ) if total_latencies else 0,
            "total_tokens": total_tokens,
            "success_rate": round(
                (total_succeeded / total_agents * 100) if total_agents else 0, 1
            ),
        }

    def generate_downloadable_report(self, user_id: str, trace_id: str = "") -> Optional[Dict[str, Any]]:
        """Generates a comprehensive downloadable report for a specific user's trace or the latest."""
        user_traces = self._get_user_traces(user_id)
        if trace_id:
            trace = next((t for t in user_traces if t["trace_id"] == trace_id), None)
        else:
            trace = user_traces[-1] if user_traces else None

        if not trace:
            return None

        return {
            "report_title": "SEED AI Simulation Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trace": trace,
            "execution_summary": self._generate_summary(trace),
            "performance_metrics": {
                "total_agents": len(trace.get("agents_invoked", [])),
                "succeeded": trace.get("agents_succeeded", 0),
                "failed": trace.get("agents_failed", 0),
                "total_latency_sec": trace.get("total_latency_sec", 0),
                "confidence": trace.get("confidence_score", "Low"),
            },
        }
