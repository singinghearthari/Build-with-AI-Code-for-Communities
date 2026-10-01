import time
import json
import logging
from typing import Dict, Any, Optional
from services.openrouter_service import OpenRouterService
from services import gemini_service
from google.genai import types

logger = logging.getLogger("HybridVisionService")


class HybridVisionService:
    _instance = None

    def __init__(self):
        self.openrouter = OpenRouterService.get_instance()

    @classmethod
    def get_instance(cls) -> "HybridVisionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_plant_disease(
        self,
        image_bytes: bytes,
        crop: str = "crop",
        conf_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        result = {"pipeline": "openrouter", "timing": {}}
        pipeline_start = time.time()

        step_start = time.time()
        or_result = self.openrouter.analyze_plant_disease(image_bytes, crop=crop)
        result["timing"]["openrouter_ms"] = or_result.get("latency_ms", 0)

        if or_result["status"] == "success":
            result["expert_analysis"] = or_result["analysis"]
            result["expert_model"] = or_result.get("model", "google/gemma-4-31b-it:free")
        else:
            logger.warning(f"OpenRouter failed: {or_result.get('message')}, falling back to Gemini")
            result["timing"]["openrouter_ms"] = round((time.time() - step_start) * 1000, 1)
            gemini_result = self._gemini_fallback(image_bytes, crop)
            result["expert_analysis"] = gemini_result.get("analysis", {})
            result["expert_model"] = f"{gemini_service.DEFAULT_MODEL} (fallback)"

        result["timing"]["total_ms"] = round((time.time() - pipeline_start) * 1000, 1)
        result["status"] = "success"
        return result

    def analyze_waste(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        pipeline_start = time.time()
        or_result = self.openrouter.analyze_plant_disease(image_bytes, crop="waste")
        return {
            "pipeline": "openrouter",
            "status": "success",
            "expert_analysis": or_result.get("analysis", {"raw_response": "Analysis unavailable"}) if or_result["status"] == "success" else {"raw_response": "Analysis unavailable"},
            "expert_model": or_result.get("model", "google/gemma-4-31b-it:free") if or_result["status"] == "success" else "unavailable",
            "timing": {"openrouter_ms": or_result.get("latency_ms", 0), "total_ms": round((time.time() - pipeline_start) * 1000, 1)},
        }

    def _gemini_fallback(self, image_bytes: bytes, crop: str) -> Dict[str, Any]:
        try:
            prompt = (
                f"You are an expert plant pathologist analyzing a {crop} leaf image. "
                "Return ONLY valid JSON:\n"
                "{\n"
                '  "disease": "disease name or Healthy",\n'
                '  "severity": "Low/Medium/High/Critical",\n'
                '  "confidence": 0-100,\n'
                '  "causes": "causes",\n'
                '  "affected_area_percent": 0-100,\n'
                '  "symptoms": "symptoms",\n'
                '  "treatment": "treatment",\n'
                '  "prevention": "prevention",\n'
                '  "organic_remedies": "organic options",\n'
                '  "immediate_action": "immediate action",\n'
                '  "visual_explanation": "visual description"\n'
                "}"
            )
            response = gemini_service.generate_with_vision(
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
                temperature=0.1,
            )
            parsed = json.loads(response.text.strip().strip("` \n"))
            return {"status": "success", "analysis": parsed}
        except Exception as e:
            logger.error(f"Gemini fallback failed: {e}")
            return {
                "status": "error",
                "analysis": {
                    "disease": "Unknown", "severity": "Unknown", "confidence": 0,
                    "visual_explanation": f"Analysis unavailable: {str(e)[:100]}",
                },
            }
