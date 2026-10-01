from typing import Dict, Any
from pydantic import BaseModel
from .base_agent import BaseAgent
from services.hybrid_vision_service import HybridVisionService
from services import gemini_service
import base64
from google.genai import types


class VisionAnalysisResult(BaseModel):
    disease: str
    severity: str
    confidence: float
    affected_area_percent: float
    visual_explanation: str
    immediate_action: str


SEVERITY_KEYWORDS: dict[str, str] = {
    "severe": "Critical", "critical": "Critical", "advanced": "High",
    "late": "High", "moderate": "Medium", "early": "Low", "mild": "Low", "healthy": "Low",
}


def _estimate_severity(label: str) -> str:
    label_lower = label.lower()
    for kw, severity in SEVERITY_KEYWORDS.items():
        if kw in label_lower:
            return severity
    return "Medium"


def _immediate_action(disease: str, severity: str) -> str:
    if disease.lower() == "healthy":
        return "No action required. Continue regular monitoring."
    if severity in ("High", "Critical"):
        return (
            f"Immediate intervention required for {disease}. "
            "Apply recommended fungicide/pesticide and isolate affected plants."
        )
    return f"Monitor {disease} progression. Apply organic treatment if symptoms worsen."


class VisionAgent(BaseAgent):

    def __init__(self):
        super().__init__("Vision")
        self.default_model = gemini_service.DEFAULT_MODEL
        self.hybrid = HybridVisionService.get_instance()

    def _process(self, context: Dict[str, Any]) -> tuple:
        image_bytes = context.get("image_bytes")
        image_base64 = context.get("image_base64")
        crop = context.get("crop", "crop")

        raw_bytes = image_bytes or (base64.b64decode(image_base64) if image_base64 else None)
        if not raw_bytes:
            raise ValueError("No image data provided")

        self.log_execution("Running OpenRouter (Gemma 4 31B) vision analysis")
        result = self.hybrid.analyze_plant_disease(raw_bytes, crop=crop)
        analysis = result.get("expert_analysis", {})
        timing = result.get("timing", {})

        disease = analysis.get("disease", "Unknown")
        severity = analysis.get("severity", "Medium")
        confidence = float(analysis.get("confidence", analysis.get("confidence_score", 50)))
        affected_area = float(analysis.get("affected_area_percent", 30))
        visual_explanation = analysis.get("visual_explanation", "")
        action = analysis.get("immediate_action", _immediate_action(disease, severity))

        vision_result = VisionAnalysisResult(
            disease=disease, severity=severity, confidence=confidence,
            affected_area_percent=affected_area, visual_explanation=visual_explanation,
            immediate_action=action,
        )

        model_used = result.get("expert_model", "unknown")
        tool_calls = [f"OpenRouter ({model_used})"]

        data = vision_result.model_dump()
        data["_hybrid"] = {
            "expert_analysis": analysis,
            "expert_model": model_used,
            "total_pipeline_ms": timing.get("total_ms", 0),
            "pipeline": result.get("pipeline", "openrouter"),
        }

        confidence_val = data.get("confidence", 50)
        reasoning = data.get("visual_explanation", "")
        return (data, tool_calls, 0, confidence_val, reasoning)
