"""
SEED AI — Budget Planning Agent (Production)
Retrieves treatment costs from knowledge base, uses Gemini to recommend plans.
Returns structured AgentResult — never fakes success.
"""
from typing import Dict, Any
from pydantic import BaseModel
from .base_agent import BaseAgent
from utils.dataset_manager import DatasetManager


class TreatmentOption(BaseModel):
    name: str = "Standard biological control"
    type: str = "Biological"
    application: str = "Foliar spray"
    cost_estimate_inr: float = 350.0
    effectiveness: str = "High"


class BudgetPlan(BaseModel):
    cheapest_option: TreatmentOption = TreatmentOption(name="Neem oil spray", cost_estimate_inr=250.0)
    best_value_option: TreatmentOption = TreatmentOption(name="Bio-fungicide", cost_estimate_inr=450.0)
    budget_limit: float = 5000.0
    estimated_total_cost: float = 450.0
    budget_compliant: bool = True
    savings_tip: str = "Opt for biological controls to save on chemical pesticide costs"
    reasoning: str = "Budget-optimized treatment strategy"


class BudgetPlanningAgent(BaseAgent):

    def __init__(self):
        super().__init__("BudgetPlanning")
        self.dataset_manager = DatasetManager()

    def _process(self, context: Dict[str, Any]) -> tuple:
        budget = context.get("budget", 0)
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
        self.log_execution(f"Planning budget for limit ₹{budget} (crop={crop}, disease={disease})")

        # Step 1: Retrieve treatment data from knowledge base
        treatments_data = self.dataset_manager.query("treatments", disease or crop)
        tool_calls = ["Knowledge Base (treatments)"]

        prompt = f"""
You are a farm budget planning advisor for Indian farmers.

Budget limit: ₹{budget}
Crop: {crop}
Disease detected: {disease}
Available treatment data from knowledge base: {treatments_data}

Based on this information, recommend:
1. The cheapest treatment option (name, type, application method, cost in INR, effectiveness)
2. The best value option (best effectiveness-to-cost ratio)
3. Whether the total cost fits within the budget
4. A practical savings tip

If no treatment data is available, use your agricultural knowledge.
All costs must be in Indian Rupees (INR).
"""
        response = self.call_llm(prompt, schema=BudgetPlan)
        result = BudgetPlan.model_validate_json(response.text)
        tokens = response.total_tokens

        return (
            result.model_dump(),
            tool_calls,
            tokens,
            80.0 if treatments_data else 55.0,
            result.reasoning,
        )
