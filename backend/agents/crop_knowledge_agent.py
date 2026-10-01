"""
SEED AI — Crop Knowledge Agent (Production)
RAG-based agent that retrieves disease/treatment info from knowledge base.
Returns structured AgentResult — never fakes success.
"""
from typing import Dict, Any
from pydantic import BaseModel
from .base_agent import BaseAgent
from utils.dataset_manager import DatasetManager


class CropKnowledgeResult(BaseModel):
    disease_info: str = "Agricultural management knowledge for crop health"
    spread_risk: str = "Moderate"
    treatment_protocols: list[str] = ["Maintain optimal field sanitation", "Apply recommended biological or chemical spray"]
    fertilizer_guidance: str = "Balanced NPK application according to soil requirements"
    best_practices: list[str] = ["Regular field monitoring", "Ensure proper spacing and drainage"]
    reasoning: str = "Evidence-based crop advisory from agricultural knowledge base"


class CropKnowledgeAgent(BaseAgent):

    def __init__(self):
        super().__init__("CropKnowledge")
        self.dataset_manager = DatasetManager()

    def _process(self, context: Dict[str, Any]) -> tuple:
        disease = context.get("disease", "")
        if not disease:
            v_res = context.get("vision_result") or {}
            disease = v_res.get("disease") or v_res.get("expert_analysis", {}).get("disease", "")
        if not disease:
            dp_res = context.get("disease_prediction_result") or {}
            preds = dp_res.get("predicted_diseases") or []
            if preds and isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                disease = preds[0].get("disease_name", "")
        if disease == "Healthy" or disease == "Unknown":
            disease = ""

        crop = context.get("crop", "")
        self.log_execution(f"Looking up knowledge for disease={disease}, crop={crop}")

        # Step 1: Retrieve from knowledge base
        disease_data = self.dataset_manager.query("diseases", disease or crop)
        treatment_data = self.dataset_manager.query("treatments", disease or crop)
        tool_calls = ["Knowledge Base (diseases)", "Knowledge Base (treatments)"]

        prompt = f"""
You are an expert agronomist and crop disease specialist.

Crop: {crop}
Detected disease: {disease}
Disease knowledge base data: {disease_data}
Treatment knowledge base data: {treatment_data}

Based on this information, provide:
1. Disease information summary
2. Spread risk assessment (Low / Medium / High / Critical)
3. Specific treatment protocols (list of actionable steps)
4. Fertilizer guidance during disease management
5. Best farming practices to prevent recurrence

Use the knowledge base data as primary source. Supplement with your expertise where data is missing.
"""
        response = self.call_llm(prompt, schema=CropKnowledgeResult)
        result = CropKnowledgeResult.model_validate_json(response.text)
        tokens = response.total_tokens

        has_kb_data = bool(disease_data or treatment_data)
        return (
            result.model_dump(),
            tool_calls,
            tokens,
            85.0 if has_kb_data else 60.0,
            result.reasoning,
        )
