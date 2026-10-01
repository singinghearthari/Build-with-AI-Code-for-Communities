"""
SEED AI — Memory Agent
Production Firestore-backed persistent memory with multiple collections.
"""
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from .base_agent import BaseAgent
from services.firebase_service import get_firestore_client, is_available as firebase_available


SCHEMA_VERSION = 2


class MemoryAgent(BaseAgent):
    """
    Production Memory Agent using Firestore.
    Manages persistent state across sessions with multiple collections:
    - farm_memory: Core farm profile and last recommendations
    - conversation_history: Past queries and responses
    - execution_logs: Agent execution traces
    - disease_history: Detected diseases over time
    - crop_history: Crop-related events
    """

    COLLECTIONS = [
        "farm_memory", "conversation_history", "execution_logs",
        "disease_history", "crop_history",
    ]

    def __init__(self):
        super().__init__("Memory")
        self.db = get_firestore_client()
        self._cache: Dict[str, dict] = {}
        if self.db:
            self.logger.info("Firestore Memory Agent initialized.")
        else:
            self.logger.warning("Firestore unavailable. Memory operations will be skipped.")

    def _get_cached_or_fetch(self, cache_key: str, fetch_fn, ttl: int = 30):
        now = time.time()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if now - entry["ts"] < ttl:
                return entry["data"]
        data = fetch_fn()
        self._cache[cache_key] = {"data": data, "ts": now}
        return data

    def _invalidate_cache(self, prefix: str):
        keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
        for k in keys_to_delete:
            del self._cache[k]

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        action = context.get("memory_action", "retrieve")
        user_id = context.get("user_id", "default_user")

        if not self.db:
            self.log_execution("Firestore unavailable, skipping memory operation")
            return {"status": "skipped", "reason": "No database connection"}

        if action == "retrieve":
            return self._retrieve_farm_memory(user_id)
        elif action == "update":
            return self._update_farm_memory(user_id, context.get("memory_data", {}))
        elif action == "log_conversation":
            return self._log_conversation(user_id, context)
        elif action == "log_execution":
            return self._log_execution_trace(user_id, context)
        elif action == "log_disease":
            return self._log_disease(user_id, context)
        elif action == "get_history":
            limit = context.get("limit", 20)
            page = context.get("page", 1)
            search = context.get("search", "")
            confidence = context.get("confidence", "")
            return self._get_conversation_history(user_id, limit, page, search, confidence)
        elif action == "get_traces":
            return self._get_execution_traces(context)
        elif action == "batch_persist":
            return self._batch_persist(user_id, context)
        elif action == "dlq_retry":
            return self._retry_dlq()

        return {"error": f"Unknown memory action: {action}"}

    def _retrieve_farm_memory(self, user_id: str) -> Dict[str, Any]:
        self.log_execution(f"Retrieving farm memory for {user_id}")
        def _fetch():
            try:
                doc = self.db.collection("farm_memory").document(user_id).get()
                if doc.exists:
                    return doc.to_dict()
                return {"status": "no_memory_found"}
            except Exception as e:
                self.logger.error(f"Memory retrieval failed: {e}")
                return {"status": "error", "error": str(e)}

        return self._get_cached_or_fetch(f"farm_memory:{user_id}", _fetch, ttl=60)

    def _update_farm_memory(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution(f"Updating farm memory for {user_id}")
        self._invalidate_cache(f"farm_memory:{user_id}")
        try:
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.db.collection("farm_memory").document(user_id).set(data, merge=True)
            return {"status": "updated"}
        except Exception as e:
            self.logger.error(f"Memory update failed: {e}")
            return {"status": "error", "error": str(e)}

    def _log_conversation(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution(f"Logging conversation for {user_id}")
        try:
            entry = {
                "user_id": user_id,
                "query": context.get("query", ""),
                "recommendation": context.get("recommendation", {}),
                "agents_used": context.get("agents_used", []),
                "confidence": context.get("confidence", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.db.collection("conversation_history").add(entry)
            return {"status": "conversation_logged"}
        except Exception as e:
            self.logger.error(f"Conversation logging failed: {e}")
            return {"status": "error", "error": str(e)}

    def _log_execution_trace(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution(f"Logging execution trace for {user_id}")
        try:
            trace = {
                "user_id": user_id,
                "trace_id": context.get("trace_id", ""),
                "agents_invoked": context.get("agents_invoked", []),
                "total_latency_sec": context.get("total_latency_sec", 0),
                "confidence": context.get("confidence", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.db.collection("execution_logs").add(trace)
            return {"status": "trace_logged"}
        except Exception as e:
            self.logger.error(f"Trace logging failed: {e}")
            return {"status": "error", "error": str(e)}

    def _log_disease(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution(f"Logging disease detection for {user_id}")
        try:
            entry = {
                "user_id": user_id,
                "disease": context.get("disease", ""),
                "crop": context.get("crop", ""),
                "severity": context.get("severity", ""),
                "confidence": context.get("confidence", 0),
                "image_path": context.get("image_path", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.db.collection("disease_history").add(entry)
            return {"status": "disease_logged"}
        except Exception as e:
            self.logger.error(f"Disease logging failed: {e}")
            return {"status": "error", "error": str(e)}

    def _batch_persist(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single atomic batch commit for all persistence needs.
        Replaces 4 individual writes with 1 Firestore Batch.
        Includes schema versioning for upgrade safety.
        Falls back to DLQ on failure.
        """
        self.log_execution(f"Batch persisting for {user_id}")
        now = datetime.now(timezone.utc).isoformat()
        try:
            batch = self.db.batch()

            # 1. Farm memory (upsert)
            farm_data = context.get("farm_memory", {})
            if farm_data:
                farm_data["_schema_version"] = SCHEMA_VERSION
                farm_data["_migrated_at"] = now
                farm_data["updated_at"] = now
                farm_ref = self.db.collection("farm_memory").document(user_id)
                batch.set(farm_ref, farm_data, merge=True)

            # 2. Conversation history
            conv_data = context.get("conversation", {})
            if conv_data:
                conv_data["_schema_version"] = SCHEMA_VERSION
                conv_data["timestamp"] = now
                conv_ref = self.db.collection("conversation_history").document()
                batch.set(conv_ref, conv_data)

            # 3. Execution trace
            trace_data = context.get("execution_trace", {})
            if trace_data:
                trace_data["_schema_version"] = SCHEMA_VERSION
                trace_data["timestamp"] = now
                trace_ref = self.db.collection("execution_logs").document()
                batch.set(trace_ref, trace_data)

            # 4. Disease history (if present)
            disease_data = context.get("disease", {})
            if disease_data:
                disease_data["_schema_version"] = SCHEMA_VERSION
                disease_data["timestamp"] = now
                disease_ref = self.db.collection("disease_history").document()
                batch.set(disease_ref, disease_data)

            batch.commit()
            self._invalidate_cache(f"traces:{user_id}")
            self._invalidate_cache(f"history:{user_id}")
            return {"status": "batch_persisted", "collections": [k for k in ["farm_memory", "conversation_history", "execution_logs", "disease_history"] if context.get(k.replace("_memory", "_memory").replace("farm_", "farm_"))]}
        except Exception as e:
            self.logger.error(f"Batch persist failed: {e}")
            self._write_dlq(user_id, context, str(e))
            return {"status": "dlq_queued", "error": str(e)}

    def _write_dlq(self, user_id: str, payload: Dict[str, Any], error: str):
        """Dead-letter queue — writes failed payloads to a DLQ collection."""
        try:
            self.db.collection("dlq_events").add({
                "user_id": user_id,
                "payload": json.dumps(payload, default=str),
                "error": error,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": 0,
            })
            self.logger.warning(f"DLQ: batch persist queued for user {user_id}")
        except Exception as dlq_err:
            self.logger.error(f"DLQ write also failed: {dlq_err}")

    def _retry_dlq(self, max_retries: int = 3) -> Dict[str, Any]:
        """Retry all pending DLQ events (called by a scheduler)."""
        try:
            docs = self.db.collection("dlq_events").where("retry_count", "<", max_retries).limit(20).stream()
            retried = 0
            for doc in docs:
                data = doc.to_dict()
                try:
                    payload = json.loads(data["payload"])
                    self._batch_persist(data["user_id"], payload)
                    doc.reference.delete()
                    retried += 1
                except Exception as e:
                    doc.reference.update({"retry_count": firestore.Increment(1), "last_error": str(e)})
            return {"status": "dlq_retry_complete", "retried": retried}
        except Exception as e:
            self.logger.error(f"DLQ retry failed: {e}")
            return {"status": "error", "error": str(e)}

    def _get_conversation_history(
        self, user_id: str, limit: int = 20, page: int = 1,
        search: str = "", confidence: str = "",
    ) -> Dict[str, Any]:
        self.log_execution(f"Fetching conversation history for {user_id}")

        def _fetch():
            offset = (page - 1) * limit
            
            query = (
                self.db.collection("conversation_history")
                .where("user_id", "==", user_id)
                .order_by("timestamp", direction="DESCENDING")
            )
            
            if confidence and confidence != "all":
                query = query.where("confidence", "==", confidence)
            
            docs = query.limit(limit).offset(offset).stream()
            results = [doc.to_dict() for doc in docs]
            
            if search:
                search_lower = search.lower()
                results = [
                    r for r in results
                    if search_lower in (r.get("query", "") or "").lower()
                ]
            
            try:
                count_query = (
                    self.db.collection("conversation_history")
                    .where("user_id", "==", user_id)
                )
                if confidence and confidence != "all":
                    count_query = count_query.where("confidence", "==", confidence)
                total = count_query.count().get()[0][0].value
            except Exception:
                total = len(results)
            
            return {"history": results, "total": total, "count": len(results)}

        try:
            return self._get_cached_or_fetch(
                f"history:{user_id}:{limit}:{page}:{search}:{confidence}",
                _fetch, ttl=120,
            )
        except Exception as e:
            self.logger.error(f"History fetch failed: {e}")
            return {"history": [], "total": 0, "error": str(e)}

    def _get_execution_traces(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context.get("user_id", "")
        limit = context.get("limit", 50)
        self.log_execution(f"Fetching execution traces for user={user_id or 'all'}")

        def _fetch():
            query = self.db.collection("execution_logs")
            if user_id:
                query = query.where("user_id", "==", user_id)
            docs = query.order_by("timestamp", direction="DESCENDING").limit(limit).stream()
            return [doc.to_dict() for doc in docs]

        try:
            cache_key = f"traces:{user_id or 'all'}:{limit}"
            traces = self._get_cached_or_fetch(cache_key, _fetch, ttl=30)
            return {"traces": traces, "count": len(traces)}
        except Exception as e:
            self.logger.error(f"Trace fetch failed: {e}")
            return {"traces": [], "error": str(e)}
